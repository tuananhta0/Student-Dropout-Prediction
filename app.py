import traceback
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from chatbot_engine import CMCChatbotEngine, row_to_clean_dict, df_to_clean_records

app = FastAPI(title="CMC Academic AI Portal V3.2.1")
engine = CMCChatbotEngine(artifact_dir="artifacts_model")

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    role: str = "student"
    logged_in_user: str = ""

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path("templates/index.html")
    if not html_path.exists():
        return HTMLResponse("<h3>Thiếu file templates/index.html</h3>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.post("/api/login")
def login(req: LoginRequest):
    u = req.username.strip().upper()
    p = req.password.strip().upper()

    if u == "ADMIN":
        if p != "ADMIN":
            raise HTTPException(status_code=401, detail="Mật khẩu ADMIN mặc định là ADMIN")
        return {
            "user_id": "ADMIN",
            "role": "admin",
            "role_label": "Quản trị Học vụ Toàn trường (19 Khoa)",
            "permissions": ["Xem thống kê toàn bộ 19 Khoa & 2.258 Lớp", "Xem toàn bộ 20.000 Sinh viên", "Tra cứu Quy chế PDF"]
        }

    if u in ["GV001", "GV002", "GV003", "ADVISOR", "COVAN"]:
        if p != u:
            raise HTTPException(status_code=401, detail=f"Mật khẩu Cố vấn mặc định là {u}")
        return {
            "user_id": u,
            "role": "advisor",
            "role_label": f"Cố vấn Học tập ({u} - Quản lý Lớp & Khoa)",
            "permissions": ["Xem số liệu tóm tắt từng Lớp & Khoa", "Xem chi tiết & Cảnh báo từng SV", "Tra cứu Quy chế PDF"]
        }

    records = engine.get_student_records(u)
    if records.empty:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy Mã SV '{u}'. Hãy thử: BIT250007, BIT250006, BIT250008, GV001 hoặc ADMIN.")
    if p != u:
        raise HTTPException(status_code=401, detail=f"Mật khẩu mặc định của sinh viên trùng với Mã SV ({u}).")

    latest = row_to_clean_dict(records.iloc[-1])
    return {
        "user_id": str(latest.get("student_id", u)),
        "role": "student",
        "role_label": f"Sinh viên Lớp {latest.get('class_code', 'IT25-18')} ({latest.get('major_name', 'CNTT')})",
        "permissions": ["Chỉ xem bảng điểm & biểu đồ của bản thân", "Tra cứu Quy chế đào tạo PDF"]
    }

@app.get("/api/student/me")
def get_student_me(user_id: str, role: str = "student"):
    records = engine.get_student_records(user_id)
    if records.empty:
        raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ sinh viên.")
    latest = row_to_clean_dict(records.iloc[-1])
    history = df_to_clean_records(records)
    warnings = engine.evaluate_student_warnings(latest)
    courses = engine.get_student_courses(str(latest.get("record_uid", "")))
    return {
        "latest": latest,
        "history": history,
        "warnings": warnings,
        "courses": courses
    }

@app.get("/api/advisor/classes")
def get_advisor_classes(role: str = "advisor", major_code: str = ""):
    if role not in ["advisor", "admin"]:
        raise HTTPException(status_code=403, detail="Chỉ Cố vấn học tập và Admin mới có quyền xem danh sách lớp.")
    return {"classes": engine.get_all_classes_summary(major_code)}

@app.get("/api/advisor/class/{class_code}")
def get_advisor_class_detail(class_code: str, role: str = "advisor"):
    if role not in ["advisor", "admin"]:
        raise HTTPException(status_code=403, detail="Không có quyền truy cập.")
    detail = engine.get_class_detail(class_code)
    if not detail:
        raise HTTPException(status_code=404, detail="Không tìm thấy lớp học.")
    return detail

@app.get("/api/admin/faculties")
def get_admin_faculties(role: str = "admin"):
    if role != "admin":
        raise HTTPException(status_code=403, detail="Chỉ Admin mới có quyền xem tổng kết 19 Khoa.")
    return {"faculties": engine.get_faculties_overview()}

@app.get("/api/admin/faculty/{major_code}")
def get_admin_faculty_detail(major_code: str, role: str = "admin"):
    if role not in ["admin", "advisor"]:
        raise HTTPException(status_code=403, detail="Không có quyền truy cập.")
    detail = engine.get_faculty_detail(major_code)
    if not detail:
        raise HTTPException(status_code=404, detail="Không tìm thấy Khoa.")
    return detail

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        return engine.answer_query(req.message, role=req.role, logged_in_user=req.logged_in_user)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=200, content={"reply": f"<div class='text-xs text-rose-400'>Lỗi xử lý câu hỏi: {str(e)}</div>", "student": None, "regulations": []})