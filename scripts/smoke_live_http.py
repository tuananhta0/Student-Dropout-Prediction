"""Kiểm thử HTTP qua Uvicorn thật ở localhost, KHÔNG thay thế kiểm tra trình duyệt.

Chỉ dùng tài khoản và dữ liệu demo. Không in token, mật khẩu, hồ sơ hay
nội dung chat; không gọi Gemini trực tuyến trong script này.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse
import httpx
from scripts.report_provenance import provenance


def probe(url='http://127.0.0.1:8000'):
    uri=urlparse(url)
    if uri.scheme!='http' or uri.hostname not in ('localhost','127.0.0.1') or uri.username or uri.password or uri.query or uri.fragment or uri.path not in ('','/'):
        raise ValueError('Chỉ cho phép http://127.0.0.1:<port> hoặc localhost')
    checks=[]
    def check(name,passed,detail=None,kind='automated'):
        checks.append({'name':name,'status':'PASS' if passed else 'FAIL','detail':detail,'kind':kind})
    with httpx.Client(base_url=url.rstrip('/'),timeout=30,trust_env=False) as c:
        homepage=c.get('/');check('Homepage',homepage.status_code==200 and 'portal.js' in homepage.text)
        accounts={}
        for account,role in [('BIT250093','student'),('GV001','advisor'),('ADMIN','admin')]:
            response=c.post('/api/login',json={'username':account,'password':account})
            ok=response.status_code==200 and response.json().get('role')==role
            check('Login '+role,ok)
            if ok: accounts[role]=response.json()['access_token']
        def headers(role):return {'Authorization':'Bearer '+accounts[role]}
        if 'student' in accounts:
            r=c.get('/api/student/me',params={'user_id':'BIT250093'},headers=headers('student'))
            check('Student own profile',r.status_code==200)
            if r.status_code==200:
                profile=r.json();check('Student course and GPA contract',all(k in profile for k in ('latest','courses','gpa_projection','courses_status')))
            r=c.get('/api/student/me',params={'user_id':'BBA250001'},headers=headers('student'))
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
            for path,payload in [('/api/student/gpa-scenario',{'user_id':'BIT250093','new_credits':3,'new_gpa':3.0,'changes':{}}),
                                 ('/api/student/payment-scenario',{'user_id':'BIT250093','payment':1000.0})]:
                r=c.post(path,json=payload,headers=headers('student'))
                check('Scenario HTTP '+path,r.status_code in (200,422),
                      'HTTP '+str(r.status_code)+' (422 cần kiểm tra điều kiện hồ sơ, không tính là nghiệp vụ PASS)',kind='transport')
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
            r=c.get('/api/student/me',params={'user_id':'BIT250093'},headers=headers('student'))
            check('Logout revokes session',r.status_code==401)
    return {'checks':checks,'passed':sum(x['status']=='PASS' for x in checks),
            'failed':sum(x['status']=='FAIL' for x in checks),
            'note':'HTTP thật nhưng không tự kiểm tra giao diện đồ họa, màn hình nhỏ hoặc xác thực ý nghĩa kịch bản 422.',
            'provenance':provenance(check='local_http_127.0.0.1')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--output',type=Path,default=Path('data/kiem_tra_ban_giao/http_local.json'))
    a=p.parse_args()
    report=probe(a.url)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'failed':report['failed'],'checks':report['checks']},ensure_ascii=False,indent=2))
    return 0 if not report['failed'] else 1

if __name__=='__main__':raise SystemExit(main())
