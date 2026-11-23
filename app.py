"""API học vụ demo: xác thực phía máy chủ và kiểm tra quyền trước khi truy cập dữ liệu."""
import logging
import re
import threading
from pathlib import Path
from typing import Literal

import bleach
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from core.chatbot_engine import CMCChatbotEngine, row_to_clean_dict, risk_display
from core.security import (SessionStore, RateLimiter, authenticate, permitted_student, require_role, require_major, require_class, scoped_classes, ADVISOR_MAJORS)
from core.runtime_config import configure_runtime
from core.retrieval import RegulationSearch, PDF, SUPPLEMENT
from core.intent_router import route
from core.data_pipeline import ROOT

app = FastAPI(title="CMC Academic AI Portal", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost','127.0.0.1','[::1]','testserver'])
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
engine = CMCChatbotEngine(artifact_dir=configure_runtime())
retrieval = RegulationSearch()
sessions = SessionStore()
bearer = HTTPBearer(auto_error=False)
contexts = {}
context_lock = threading.Lock()
chat_busy=set()
login_limit=RateLimiter(30,60)
chat_limit=RateLimiter(15,60)
logger = logging.getLogger(__name__)

# Chỉ demo localhost. Host header không thay thế kiểm tra địa chỉ kết nối.
@app.middleware('http')
async def local_only(request: Request, call_next):
    if request.client is None or request.client.host not in ('127.0.0.1','::1','testclient'):
        return JSONResponse(status_code=403,content={'detail':'Chỉ cho phép truy cập localhost trong bản mô phỏng'})
    response = await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='no-referrer'
    response.headers['X-Frame-Options']='DENY'
    response.headers['Cache-Control']='no-store'
    # JavaScript cục bộ, không inline; style động của biểu đồ vẫn cần inline.
    response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    return response

@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    return JSONResponse(status_code=422,content={'detail':'Dữ liệu yêu cầu không hợp lệ'})

@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    logger.error('Lỗi nội bộ tại API %s: %s',request.url.path,type(exc).__name__)
    return JSONResponse(status_code=500,content={'detail':'Không thể xử lý yêu cầu lúc này'})


def credential(auth: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if auth is None: raise HTTPException(401,'Cần đăng nhập',headers={'WWW-Authenticate':'Bearer'})
    return auth.credentials


def principal(token: str = Depends(credential)):
    return sessions.resolve(token)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1,max_length=64)
    password: str = Field(min_length=1,max_length=128)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1,max_length=2000)

@app.get('/', response_class=HTMLResponse)
def index():
    return HTMLResponse((ROOT/'templates/index.html').read_text(encoding='utf-8'))

@app.get('/regulations/'+PDF)
def regulation_pdf():
    return FileResponse(ROOT/'regulations'/PDF,media_type='application/pdf')

@app.get('/regulations/'+SUPPLEMENT)
def regulation_supplement():
    return FileResponse(ROOT/'regulations'/SUPPLEMENT,media_type='application/pdf')

@app.get('/favicon.ico',include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get('/api/data-view')
def data_view(p=Depends(principal)):
    # Sinh viên chỉ cần mốc dữ liệu và phiên bản, không cần số liệu toàn trường.
    base={'mode':engine.data_mode,'as_of':engine.data_as_of,'synthetic':True,'model_version':engine.meta.get('version','baseline')}
    if p.role=='admin':
        base.update(statistics_scope='Sinh viên đã có kết quả học kỳ tại mốc dữ liệu',snapshots=len(engine.ml_df),academic_terms=len(engine.df),active_students=len(engine.active_latest_df),students_with_snapshots=len(engine.latest_df),valid_predictions=int(engine.active_latest_df.prediction_status.eq('VALID').sum()))
    return base

@app.post('/api/login')
def login(req: LoginRequest,request: Request):
    login_limit.check(request.client.host)
    p=authenticate(engine,req.username,req.password)
    if p.role=='student':
        row=engine.get_student_records(p.user_id).iloc[-1]
        label=f"Sinh viên Lớp {row['class_code']} ({row['major_name']})"
    elif p.role=='advisor': label='Cố vấn học tập (phạm vi mô phỏng: '+', '.join(sorted(ADVISOR_MAJORS[p.user_id]))+')'
    else: label='Quản trị học vụ toàn trường'
    token=sessions.issue(p)
    with context_lock:
        # Dọn các phiên đã hết hạn/đăng xuất; không lưu nội dung hội thoại.
        with sessions.lock:
            for k in list(contexts):
                if k not in sessions.items: contexts.pop(k,None)
        contexts[token]={}
    return {'user_id':p.user_id,'role':p.role,'role_label':label,'access_token':token,'token_type':'bearer','expires_in':sessions.ttl}

@app.post('/api/logout')
def logout(token: str=Depends(credential)):
    sessions.revoke(token)
    with context_lock: contexts.pop(token,None)
    return {'message':'Đã đăng xuất'}

# Danh sách trường phục vụ giao diện; không trả toàn bộ hàng dữ liệu ML.
PROFILE_FIELDS=set('student_id class_code major_name major_code degree_type education_mode snapshot_term_index snapshot_cutoff state_as_of data_as_of graduation_review_date prediction_cutoff prediction_status gpa4_term gpa4_cumulative credits_earned_cumulative credits_required new_course_credits_available registration_max_credits credits_failed_term failed_credit_ratio credits_owed_cumulative attendance_rate_term tuition_debt_end_vnd'.split())
COURSE_FIELDS=set('term course_id short_code course_name credits process_score exam_score final_score grade_letter is_passed exam_eligible eligibility_review grade_policy_review gpa_eligible'.split())

def public_fields(row, fields):
    return {k:v for k,v in row.items() if k in fields}

@app.get('/api/student/me')
def get_student_me(user_id: str, p=Depends(principal)):
    records=permitted_student(engine,p,user_id)
    latest=row_to_clean_dict(records.iloc[-1]); courses=engine.get_student_courses(str(latest['record_uid']))
    return {'latest':public_fields(latest,PROFILE_FIELDS),'history':[public_fields(row_to_clean_dict(r),PROFILE_FIELDS) for _,r in records.iterrows()],
            'risk_display':risk_display(latest,engine.meta['threshold']), 'warnings':engine.evaluate_student_warnings(latest),
            'gpa_projection':engine.calc_next_term_gpa_projection(latest),'research_model':bool(engine.meta.get('model_available_as_of')),
            'courses':[public_fields(c,COURSE_FIELDS) for c in courses], 'courses_status':'unavailable' if engine.df_attempts is None else 'available' if courses else 'no_records'}

class GpaScenarioRequest(BaseModel):
    user_id: str = Field(min_length=1,max_length=64)
    changes: dict[str, Literal['F','D','D+','C','C+','B','B+','A','A+']] = Field(default_factory=dict, max_length=30)
    new_gpa: float = Field(default=0, ge=0, le=4, allow_inf_nan=False, strict=True)
    new_credits: int = Field(default=0, ge=0, le=30, strict=True)


class PaymentScenarioRequest(BaseModel):
    user_id: str = Field(min_length=1,max_length=64)
    payment: float = Field(gt=0, allow_inf_nan=False, strict=True)



def scenario_uid(p,user_id):
    return str(permitted_student(engine,p,user_id).iloc[-1]['record_uid'])

@app.post('/api/student/gpa-scenario')
def gpa_scenario(req: GpaScenarioRequest,p=Depends(principal)):
    uid=scenario_uid(p,req.user_id)
    try: return engine.calculate_retake(uid,req.changes,req.new_gpa,req.new_credits)
    except (ValueError,TypeError) as exc: raise HTTPException(422,str(exc)) from exc

@app.post('/api/student/payment-scenario')
def payment_scenario_endpoint(req: PaymentScenarioRequest,p=Depends(principal)):
    uid=scenario_uid(p,req.user_id)
    try: return engine.calculate_payment_scenario(uid,req.payment)
    except (ValueError,TypeError) as exc: raise HTTPException(422,str(exc)) from exc

@app.get('/api/advisor/classes')
def get_advisor_classes(major_code: str='',p=Depends(principal)):
    require_role(p,'advisor','admin')
    if major_code: require_major(p,major_code)
    rows=engine.get_all_classes_summary(major_code)
    rows=scoped_classes(p,rows)
    return {'classes':rows}

@app.get('/api/advisor/class/{class_code}')
def get_advisor_class_detail(class_code: str,p=Depends(principal)):
    require_role(p,'advisor','admin')
    if not re.fullmatch(r'[A-Z]{2}\d{2}-\d{2,3}',class_code.upper()): raise HTTPException(422,'Mã lớp không hợp lệ')
    require_class(p,class_code.upper())
    detail=engine.get_class_detail(class_code)
    if not detail: raise HTTPException(404,'Không tìm thấy lớp')
    require_major(p,detail['major_code'])
    # Chỉ các trường thực sự hiển thị trong bảng lớp.
    allowed=set('student_id snapshot_term_index gpa4_term gpa4_cumulative credits_earned_cumulative credits_owed_cumulative attendance_rate_term prediction_status risk_display gpa_projection'.split())
    detail['students']=[public_fields(r,allowed) for r in detail['students']]
    return detail

@app.get('/api/admin/faculties')
def get_admin_faculties(p=Depends(principal)):
    require_role(p,'admin')
    return {'faculties':engine.get_faculties_overview()}

@app.get('/api/admin/faculty/{major_code}')
def get_admin_faculty_detail(major_code: str,p=Depends(principal)):
    require_role(p,'admin')
    detail=engine.get_faculty_detail(major_code)
    if not detail: raise HTTPException(404,'Không tìm thấy khoa')
    return detail

# HTML sinh bởi các dịch vụ cũ được lọc; không cho script, event handler, style hoặc link tùy ý.
def safe_link(tag,name,value):
    if name=='class': return True
    if tag=='a' and name=='href': return any(re.fullmatch(r'/regulations/'+re.escape(doc)+r'#page=\d{1,2}',value) for doc in (PDF,SUPPLEMENT))
    if tag=='a' and name=='target': return value=='_blank'
    if tag=='a' and name=='rel': return value=='noopener noreferrer'
    return tag=='td' and name=='colspan' and value.isdigit()

def sanitize_reply(text):
    return bleach.clean(text,tags={'div','span','p','b','strong','em','i','br','ul','ol','li','table','thead','tbody','tr','th','td','code','pre','a','h3','h4'},attributes=safe_link,strip=True)

@app.post('/api/chat')
def chat_endpoint(req: ChatRequest, token: str=Depends(credential)):
    p=sessions.resolve(token)
    chat_limit.check(p.user_id)
    with context_lock:
        if token in chat_busy: raise HTTPException(409,'Đang xử lý câu hỏi trước')
        chat_busy.add(token)
        context=dict(contexts.get(token,{}))
    try:
        result=route(engine,retrieval,p,req.message,context)
        sessions.resolve(token)  # Không trả kết quả nếu đã đăng xuất.
        with context_lock:
            if token in contexts: contexts[token]=context
    finally:
        with context_lock: chat_busy.discard(token)
    # Không trả hồ sơ thô hoặc kết quả công cụ ngoài phạm vi đã cấp.
    return {'reply':sanitize_reply(result['reply']),'regulations':result.get('regulations',[]),'retrieved_articles':[c['article'] for c in result.get('retrieved',result.get('regulations',[]))],'intent':result['intent'],'method':result['method'],'embedding_used':result.get('embedding_used',False),'retrieval_method':result.get('retrieval_method'),'generation_status':result.get('generation_status','NOT_REQUESTED')}
