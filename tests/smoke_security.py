"""Kiểm tra tích hợp HTTP trên model và dữ liệu được chọn. Tạo tài khoản tạm riêng để không dùng mật khẩu thật."""
import os
from pathlib import Path
import tempfile
from fastapi.testclient import TestClient
from scripts.create_accounts import create


def run():
    from app import app, engine
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'accounts.json';old=os.environ.get('CMC_ACCOUNTS_FILE')
        outsider=engine.latest_df.loc[engine.latest_df.major_code.ne('IT'),'student_id'].iloc[0]
        assigned=str(engine.latest_df.loc[engine.latest_df.student_id.eq('BIT250093'),'class_code'].iloc[0])
        otherclass=str(engine.active_latest_df.loc[engine.active_latest_df.major_code.ne('IT'),'class_code'].iloc[0])
        passwords=create(path,['ADMIN','GV001','BIT250093',str(outsider)])
        os.environ['CMC_ACCOUNTS_FILE']=str(path)
        try:
            with TestClient(app,raise_server_exceptions=False) as c:
                def login(uid):
                    r=c.post('/api/login',json={'username':uid,'password':passwords[uid]});assert r.status_code==200,r.text
                    return {'Authorization':'Bearer '+r.json()['access_token']}
                def chat(h,q,code=200):
                    r=c.post('/api/chat',headers=h,json={'message':q,'role':'admin','logged_in_user':'ADMIN'});assert r.status_code==code,(q,r.status_code,r.text[:200]);return r.json()
                assert c.get('/api/student/me',params={'user_id':'BIT250093'}).status_code==401
                assert c.post('/api/login',json={'username':'ADMIN','password':'ADMIN'}).status_code==401
                s=login('BIT250093');g=login('GV001');a=login('ADMIN')
                profile=c.get('/api/student/me',headers=s,params={'user_id':'BIT250093'});assert profile.status_code==200,profile.text[:300]
                assert not {'record_uid','gender','home_province','admission_score'} & set(profile.json()['latest'])
                assert c.get('/api/student/me',headers=s,params={'user_id':outsider,'role':'admin'}).status_code==403
                assert c.get('/api/admin/faculties',headers=s).status_code==403
                assert c.get('/api/admin/faculties',headers=a).status_code==200
                classes=c.get('/api/advisor/classes',headers=g);assert classes.status_code==200 and all(r['major_code']=='IT' for r in classes.json()['classes'])
                assert c.get('/api/advisor/class/'+otherclass,headers=g).status_code==403
                if assigned in {r['class_code'] for r in classes.json()['classes']}:assert c.get('/api/advisor/class/'+assigned,headers=g).status_code==200
                assert c.get('/api/student/me',headers=g,params={'user_id':outsider}).status_code==403
                for route in ('/api/student/gpa-scenario','/api/student/payment-scenario'):
                    assert c.post(route,headers=s,json={'user_id':outsider,'payment':1}).status_code==403
                assert chat(s,'Điểm đầu vào của em')['intent']=='admission'
                assert chat(s,'Nguy cơ thôi học của em')['intent']=='risk'
                assert chat(s,'Điều 18')['regulations']
                assert [x['id'] for x in chat(s,'Khoản 2 thì sao?')['regulations']]==['D18.K2']
                assert not chat(g,'Khoản 2 thì sao?')['regulations']
                chat(s,'Lớp mạnh nhất',403);chat(s,'Xem hồ sơ '+str(outsider),403)
                assert '<script>' not in chat(s,'<script>alert(1)</script>')['reply']
                assert c.post('/api/logout',headers=s).status_code==200
                assert c.get('/api/data-view',headers=s).status_code==401
                print('[ĐẠT] Xác thực, phân quyền 3 vai trò, RAG, quyền hồ sơ và thu hồi phiên')
        finally:
            if old is None:os.environ.pop('CMC_ACCOUNTS_FILE',None)
            else:os.environ['CMC_ACCOUNTS_FILE']=old

if __name__=='__main__':run()
