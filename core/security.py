"""Phiên demo có thời hạn, xác thực và phân quyền độc lập với client."""
from dataclasses import dataclass
import hmac
import json
import os
from pathlib import Path
import secrets
import threading
import time
from collections import deque

class RateLimiter:
    """Giới hạn cửa sổ trượt theo khóa, dung lượng hữu hạn."""
    def __init__(self, limit, window, capacity=4096):
        self.limit=limit; self.window=window; self.capacity=capacity; self.items={}; self.lock=threading.Lock()
    def check(self,key):
        with self.lock:
            now=time.monotonic()
            for k,values in list(self.items.items()):
                while values and values[0]<=now-self.window: values.popleft()
                if not values: self.items.pop(k,None)
            if key not in self.items and len(self.items)>=self.capacity: raise HTTPException(429,"Quá nhiều yêu cầu")
            values=self.items.setdefault(key,deque())
            if len(values)>=self.limit: raise HTTPException(429,"Quá nhiều yêu cầu; thử lại sau",headers={"Retry-After":str(max(1,int(self.window-(now-values[0])+1)))})
            values.append(now)
from fastapi import HTTPException

# Phân công mô phỏng cố định theo ngành; không phải phân công thật của CMC.
ADVISOR_MAJORS = {'GV001': {'IT'}, 'GV002': {'BA'}, 'GV003': {'AI'}, 'ADVISOR': {'IT'}, 'COVAN': {'IT'}}

@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str

class SessionStore:
    def __init__(self, ttl=3600, capacity=2048):
        self.ttl=ttl; self.capacity=capacity; self.items={}; self.lock=threading.Lock()
    def issue(self, principal):
        with self.lock:
            now=time.monotonic()
            self.items={k:v for k,v in self.items.items() if v[1]>now}
            if len(self.items)>=self.capacity: raise HTTPException(503,'Quá nhiều phiên đăng nhập')
            token=secrets.token_urlsafe(32)
            self.items[token]=(principal,now+self.ttl)
            return token
    def resolve(self, token):
        with self.lock:
            item=self.items.get(token)
            if item and item[1]>time.monotonic(): return item[0]
            self.items.pop(token,None)
        raise HTTPException(401,'Phiên đăng nhập không hợp lệ hoặc hết hạn')
    def revoke(self, token):
        with self.lock: self.items.pop(token,None)


def check_password(username, password):
    """Demo localhost: mật khẩu đúng bằng mã tài khoản đã chuẩn hóa.

    Không đọc file mật khẩu cũ, không tạo tài khoản hoặc hash. Hàm
    authenticate vẫn kiểm tra mã tồn tại và xác định vai trò phía server.
    Đây không phải xác thực an toàn cho dữ liệu thật.
    """
    return bool(username) and hmac.compare_digest(username.encode('utf-8'), password.encode('utf-8'))


def authenticate(engine, username, password):
    u=username.strip().upper()
    if not check_password(u,password):
        raise HTTPException(401,'Tài khoản hoặc mật khẩu không đúng')
    if u=='ADMIN': return Principal(u,'admin')
    if u in ADVISOR_MAJORS: return Principal(u,'advisor')
    r=engine.get_student_records(u)
    if r.empty or str(r.iloc[-1]['student_id']) != u:
        raise HTTPException(401,'Tài khoản hoặc mật khẩu không đúng')
    return Principal(u,'student')


def require_role(p, *roles):
    if p.role not in roles: raise HTTPException(403,'Không có quyền truy cập')


def require_major(p, major):
    require_role(p,'advisor','admin')
    if p.role=='advisor' and major.upper() not in ADVISOR_MAJORS[p.user_id]:
        raise HTTPException(403,'Ngành ngoài phạm vi cố vấn được phân công')


def assigned_classes(p):
    path=os.getenv('CMC_ADVISOR_SCOPE_FILE')
    if p.role!='advisor' or not path: return None
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    values=data.get(p.user_id,[])
    if not isinstance(values,list) or any(not isinstance(v,str) for v in values): raise HTTPException(503,'Cấu hình lớp không hợp lệ')
    return set(v.upper() for v in values)


def require_class(p,code):
    require_major(p,code[:2])
    allowed=assigned_classes(p)
    if allowed is not None and code.upper() not in allowed: raise HTTPException(403,'Lớp ngoài phạm vi cố vấn')


def scoped_classes(p,rows):
    if p.role!='advisor': return rows
    allowed=assigned_classes(p)
    return [r for r in rows if r['major_code'] in ADVISOR_MAJORS[p.user_id] and (allowed is None or r['class_code'] in allowed)]


def scoped_frame(p,frame):
    if p.role!='advisor': return frame
    out=frame[frame.major_code.isin(ADVISOR_MAJORS[p.user_id])]
    allowed=assigned_classes(p)
    return out if allowed is None else out[out.class_code.isin(allowed)]


def permitted_student(engine,p,target):
    if p.role=='student' and target.strip().upper()!=p.user_id:
        raise HTTPException(403,'Chỉ được xem hồ sơ của bản thân')
    r=engine.get_student_records(target.strip().upper())
    if r.empty: raise HTTPException(404,'Không tìm thấy hồ sơ sinh viên')
    if p.role=='advisor':
        require_major(p,str(r.iloc[-1]['major_code']))
        require_class(p,str(r.iloc[-1]['class_code']))
    return r
