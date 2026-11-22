"""Tạo mật khẩu riêng cho tài khoản demo; lưu hash, không đưa thông tin đăng nhập lên Git."""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
from core.data_pipeline import ROOT
from core.security import hash_password, ADVISOR_MAJORS


def create(path, ids):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    existing=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    added={}
    for uid in dict.fromkeys(x.upper() for x in ids):
        if uid not in ADVISOR_MAJORS and uid!='ADMIN' and not re.fullmatch(r'B[A-Z]{2}\d{6}',uid): raise ValueError('Mã tài khoản không hợp lệ')
        if uid in existing: continue
        password=secrets.token_urlsafe(16);salt=secrets.token_hex(16)
        existing[uid]={'salt':salt,'hash':hash_password(password,salt)};added[uid]=password
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:json.dump(existing,f,indent=2)
    os.replace(tmp,path)
    return added

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--student',action='append',default=[]);ap.add_argument('--output',type=Path,default=Path(os.getenv('CMC_ACCOUNTS_FILE', ROOT/'data/accounts.json')));a=ap.parse_args()
    result=create(a.output,['ADMIN','GV001','GV002','GV003','BIT250093','BIT250006','BIT250007',*a.student])
    print('Mật khẩu mới chỉ hiển thị một lần. Lưu riêng, không commit hoặc chia sẻ công khai.')
    for uid,password in result.items():print(uid,':',password)
    if not result:print('Không có tài khoản mới; giữ nguyên mật khẩu hiện có.')
