"""Kiểm thử HTTP qua Uvicorn thật ở localhost, KHÔNG thay thế kiểm tra trình duyệt.

Chỉ dùng tài khoản và dữ liệu demo. Không in token, mật khẩu, hồ sơ hay
nội dung chat; không gọi Gemini trực tuyến trong script này.
"""
import argparse
import math
import re
import json
from pathlib import Path
from urllib.parse import urlparse
import httpx
from scripts.report_provenance import provenance


def scenario_valid(kind, status, body):
    """Require business success; 422 is never silently accepted as PASS."""
    if status != 200 or not isinstance(body, dict):
        return False
    if kind == 'gpa':
        values = (body.get('gpa_before'), body.get('gpa_after'))
        return all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 4 for x in values)
    if kind == 'payment':
        before, after = body.get('before'), body.get('after')
        diff = body.get('difference_pp')
        if not isinstance(before, dict) or not isinstance(after, dict):
            return False
        probs = (before.get('risk_probability'), after.get('risk_probability'))
        return (all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in probs)
                and type(diff) in (int, float) and math.isfinite(diff))
    raise ValueError('Unknown scenario kind')


def probe(url='http://127.0.0.1:8000', student='BIT250093', payment=1000.0):
    uri=urlparse(url)
    if uri.scheme!='http' or uri.hostname not in ('localhost','127.0.0.1') or uri.username or uri.password or uri.query or uri.fragment or uri.path not in ('','/'):
        raise ValueError('Chỉ cho phép http://127.0.0.1:<port> hoặc localhost')
    if not re.fullmatch(r'B[A-Z]{2}\d{6}', student):
        raise ValueError('Student account must be a demo ID such as BIT250093')
    if not isinstance(payment, (int, float)) or not math.isfinite(payment) or payment <= 0:
        raise ValueError('Payment must be a positive finite number')
    checks=[]
    def check(name,passed,detail=None,kind='automated'):
        checks.append({'name':name,'status':'PASS' if passed else 'FAIL','detail':detail,'kind':kind})
    with httpx.Client(base_url=url.rstrip('/'),timeout=30,trust_env=False) as c:
        homepage=c.get('/');check('Homepage',homepage.status_code==200 and 'portal.js' in homepage.text)
        accounts={}
        for account,role in [(student,'student'),('GV001','advisor'),('ADMIN','admin')]:
            response=c.post('/api/login',json={'username':account,'password':account})
            ok=response.status_code==200 and response.json().get('role')==role
            check('Login '+role,ok)
            if ok: accounts[role]=response.json()['access_token']
        def headers(role):return {'Authorization':'Bearer '+accounts[role]}
        if 'student' in accounts:
            r=c.get('/api/student/me',params={'user_id':student},headers=headers('student'))
            check('Student own profile',r.status_code==200)
            if r.status_code==200:
                profile=r.json();check('Student course and GPA contract',all(k in profile for k in ('latest','courses','gpa_projection','courses_status')))
            other='BBA250001' if student!='BBA250001' else 'BIT250093'
            r=c.get('/api/student/me',params={'user_id':other},headers=headers('student'))
            check('Student cannot access other profile',r.status_code==403)
            r=c.get('/api/admin/faculties',headers=headers('student'))
            check('Student blocked from admin',r.status_code==403)
            for question,expected in [
                ('Em đang bệnh nên muốn nghỉ học một kỳ, thủ tục thế nào?',21),
                ('Khi nào đóng học phí kỳ tới?',None),
                ('Có thể chuyển đến trường khác để học không?',22)]:
                r=c.post('/api/chat',headers=headers('student'),json={'message':question})
                ok=r.status_code==200
                articles=set()
                if ok: articles={x.get('article') for x in r.json().get('regulations',[])}
                check('RAG '+question[:35],ok and (expected in articles if expected is not None else not articles))
            # Chọn kịch bản GPA không đổi để tránh vượt tín chỉ khi hồ sơ
            # có quá ít tín chỉ còn lại. Với payment, chọn SV còn nợ và có
            # prediction_status VALID. Nếu không đủ điều kiện -> báo FAIL,
            # không biến HTTP 422 thành chứng cứ nghiệp vụ thành công.
            for path,kind,payload in [
                ('/api/student/gpa-scenario','gpa',{'user_id':student,'new_credits':0,'new_gpa':0.0,'changes':{}}),
                ('/api/student/payment-scenario','payment',{'user_id':student,'payment':float(payment)}),
            ]:
                r=c.post(path,json=payload,headers=headers('student'))
                body=r.json() if r.headers.get('content-type','').startswith('application/json') else None
                ok=scenario_valid(kind,r.status_code,body)
                check('Scenario business '+kind,ok,
                      'HTTP '+str(r.status_code)+(' (choose an eligible enrolled student, with debt and VALID prediction)' if not ok else ' (validated business response)'))
        if 'advisor' in accounts:
            r=c.get('/api/advisor/classes',headers=headers('advisor'))
            check('Advisor authorized list',r.status_code==200)
            r=c.get('/api/admin/faculties',headers=headers('advisor'))
            check('Advisor blocked from admin',r.status_code==403)
        if 'admin' in accounts:
            r=c.get('/api/admin/faculties',headers=headers('admin'))
            check('Admin faculties',r.status_code==200)
        if 'student' in accounts:
            c.post('/api/logout',headers=headers('student'))
            r=c.get('/api/student/me',params={'user_id':student},headers=headers('student'))
            check('Logout revokes session',r.status_code==401)
    return {'checks':checks,'passed':sum(x['status']=='PASS' for x in checks),
            'failed':sum(x['status']=='FAIL' for x in checks),
            'note':'HTTP thật; cả hai scenario phải trả 200 với dữ liệu đúng schema. Không thay thế kiểm tra trình duyệt hoặc kiểm định tác động nhân quả.',
            'provenance':provenance(check='local_http_127.0.0.1')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--student',default='BIT250093',help='Demo student with course records, outstanding debt and VALID prediction')
    p.add_argument('--payment',type=float,default=1000.0,help='Payment amount strictly greater than zero')
    p.add_argument('--output',type=Path,default=Path('data/kiem_tra_ban_giao/http_local.json'))
    a=p.parse_args()
    report=probe(a.url,a.student,a.payment)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'failed':report['failed'],'checks':report['checks']},ensure_ascii=False,indent=2))
    return 0 if not report['failed'] else 1

if __name__=='__main__':raise SystemExit(main())
