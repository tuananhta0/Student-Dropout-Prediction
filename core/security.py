"""Phiên demo có thời hạn, xác thực và phân quyền độc lập với client."""
from dataclasses import dataclass
import hmac
import hashlib
import json
import os
from pathlib import Path
from core.data_pipeline import ROOT
import secrets
import threading
import time
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


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac('sha256',password.encode(),bytes.fromhex(salt),600000).hex()


def check_password(username, password):
    path=Path(os.getenv('CMC_ACCOUNTS_FILE',ROOT/'data/accounts.json'))
    if not path.is_file(): raise HTTPException(503,'Chưa tạo tài khoản demo; chạy scripts.create_accounts')
    accounts=json.loads(path.read_text(encoding='utf-8'))
    entry=accounts.get(username)
    # Tính hash cả khi không có tài khoản để giảm khác biệt thời gian.
    salt=entry['salt'] if entry else '00'*16
    actual=hash_password(password,salt)
    expected=entry['hash'] if entry else '00'*32
    return bool(entry) and hmac.compare_digest(actual,expected)


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


def permitted_student(engine,p,target):
    if p.role=='student' and target.strip().upper()!=p.user_id:
        raise HTTPException(403,'Chỉ được xem hồ sơ của bản thân')
    r=engine.get_student_records(target.strip().upper())
    if r.empty: raise HTTPException(404,'Không tìm thấy hồ sơ sinh viên')
    if p.role=='advisor': require_major(p,str(r.iloc[-1]['major_code']))
    return r
