from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
from chatbot_engine import CMCChatbotEngine

#khoi tao app
app = FastAPI(title="CMC University Multi-Tier AI Portal V3.2")

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

if Path("regulations").exists():
    app.mount("/regulations", StaticFiles(directory="regulations"), name="regulations")

engine = CMCChatbotEngine("artifacts_model")

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    role: str = "student"
    user_id: str = ""

@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("templates/index.html").read_text(encoding="utf-8")

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

    latest = records.iloc[-1]
    return {
        "user_id": str(latest["student_id"]),
        "role": "student",
        "role_label": f"Sinh viên Lớp {latest.get('class_code', '')} ({latest.get('major_name', '')})",
        "permissions": ["Chỉ xem bảng điểm & biểu đồ của bản thân", "Tra cứu Quy chế đào tạo PDF"]
    }
@app.get("/api/student-self/{student_id}")
async def get_student_self(student_id: str):
    records = engine.get_student_records(student_id)
    if records.empty:
        raise HTTPException(status_code=404, detail="Không tìm thấy sinh viên")
    latest_row = records.iloc[-1].fillna(0).to_dict()
    uid = str(latest_row.get("record_uid", ""))
    profile = {
        "student_id": str(latest_row.get("student_id", "")),
        "class_code": str(latest_row.get("class_code", "")),
        "major_name": str(latest_row.get("major_name", "")),
        "degree_type": str(latest_row.get("degree_type", "")),
        "education_mode": str(latest_row.get("education_mode", "")),
        "cohort_year": int(latest_row.get("cohort_year", 2025)),
        "home_province": str(latest_row.get("home_province", "")),
        "credits_earned": int(latest_row.get("credits_earned_cumulative", 0)),
        "credits_required": int(latest_row.get("credits_required", 150)),
        "failed_ratio": round(float(latest_row.get("failed_credit_ratio", 0)), 3),
        "attendance": round(float(latest_row.get("attendance_rate_term", 0)), 1)
    }
    warnings = engine.evaluate_student_warnings(latest_row)
    courses = engine.get_student_courses(uid)
    history = []
    for _, r in records.iterrows():
        history.append({
            "snapshot_term_index": int(r.get("snapshot_term_index", 1)),
            "gpa4_term": round(float(r.get("gpa4_term", 0)), 2),
            "gpa4_cumulative": round(float(r.get("gpa4_cumulative", 0)), 2),
            "credits_registered_term": int(r.get("credits_registered_term", 0)),
            "credits_failed_term": int(r.get("credits_failed_term", 0)),
            "failed_credit_ratio": round(float(r.get("failed_credit_ratio", 0)), 3),
            "credits_earned_cumulative": int(r.get("credits_earned_cumulative", 0)),
            "credits_owed_cumulative": int(r.get("credits_owed_cumulative", 0)),
            "warning_streak": int(r.get("warning_streak", 0)),
            "attendance_rate_term": round(float(r.get("attendance_rate_term", 0)), 1),
            "tuition_debt_end_vnd": float(r.get("tuition_debt_end_vnd", 0))
        })
    return {"profile": profile, "warnings": warnings, "history": history, "courses": courses}

@app.get("/api/advisor/classes")
async def get_classes(major_code: Optional[str] = ""):
    classes = engine.get_all_classes_summary(major_code or "")
    return {"total_classes": len(classes), "classes": classes[:200]}

@app.get("/api/advisor/class/{class_code}")
async def get_class_detail(class_code: str):
    detail = engine.get_class_detail(class_code)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy lớp {class_code}")
    return detail

@app.get("/api/admin/faculties")
async def get_faculties():
    facs = engine.get_faculties_overview()
    return {
        "total_faculties": len(facs),
        "total_students": int(len(engine.latest_df)),
        "total_classes": int(engine.latest_df['class_code'].nunique()),
        "faculties": facs
    }

@app.get("/api/admin/faculty/{major_code}")
async def get_faculty_detail(major_code: str):
    detail = engine.get_faculty_detail(major_code)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy khoa {major_code}")
    return detail

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    return engine.answer_query(req.message, req.role, req.user_id)