import re
from core.academic_rules import number, whole, rounded, registration_bounds, warning_reasons, withdrawal_reasons, exam_barred, course_grade, projection, graduation_conditions, retake_projection, strict_bool
import os
from core.data_pipeline import Dataset, runtime_view, visible_attempts, academic_view, digest
from core.prediction import model_inputs, risk_level, apply_prediction_scope, payment_scenario
import json
import math
import unicodedata
import joblib
import logging
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import pandas as pd
from pathlib import Path

def risk_display(row, threshold):
    """Nhãn tiếng Việt; không coi dự báo lịch sử là nguy cơ hiện tại."""
    risk = number(row.get('risk_probability'),0,1)
    if row.get('state_as_of') not in (None,'ENROLLED'):
        return 'Hồ sơ lịch sử'
    if risk is None:
        return 'Chưa có dự báo'
    if row.get('prediction_status') != 'VALID':
        return {'OUT_OF_SCOPE':'Ngoài phạm vi mô hình','RETROSPECTIVE':'Dự báo hồi cứu'}.get(row.get('prediction_status'),'Dự báo đã hết hạn')
    level = risk_level(risk, threshold)
    return f'Nguy cơ thôi học: {risk*100:.1f}% ({level}) trong 2 học kỳ tiếp theo'


def normalize_vn(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip().lower()

def strip_accents(text: str) -> str:
    text = normalize_vn(text).replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

def row_to_clean_dict(row) -> dict:
    raw = json.loads(row.to_json())
    str_cols = {"student_id", "record_uid", "class_code", "major_code", "major_name", "degree_type", "education_mode", "risk_level", "verified_certificates", "exit_reason_for_evaluation"}
    return {k: ("" if k in str_cols else None) if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else v for k, v in raw.items()}

def df_to_clean_records(df: pd.DataFrame) -> list:
    return [row_to_clean_dict(r) for _, r in df.iterrows()]

MAJOR_ALIASES = {
    "IT": ["cong nghe thong tin", "cntt", "khoa it", "nganh it"], "CS": ["khoa hoc may tinh", "khmt", "khoa cs", "nganh cs"],
    "SE": ["ky thuat phan mem", "ktpm", "khoa se", "nganh se"], "AI": ["tri tue nhan tao", "ttnt", "khoa ai", "nganh ai"],
    "NS": ["an ninh mang", "anm", "antoan thong tin", "khoa ns"], "DA": ["phan tich du lieu", "khoa hoc du lieu", "khoa da"],
    "EC": ["dien tu vien thong", "khoa ec"], "GA": ["cong nghe game", "khoa ga"],
    "BA": ["quan tri kinh doanh", "qtkd", "khoa ba", "nganh ba"], "EM": ["thuong mai dien tu", "tmdt", "khoa em"],
    "LS": ["logistics", "chuoi cung ung", "khoa ls"], "IB": ["kinh doanh quoc te", "kdqt", "khoa ib"],
    "MK": ["marketing", "khoa mk"], "CB": ["tai chinh ngan hang", "khoa cb"], "MC": ["truyen thong da phuong tien", "truyen thong", "khoa mc"],
    "PR": ["quan he cong chung", "khoa pr"], "GD": ["thiet ke do hoa", "tkdh", "khoa gd"],
    "KL": ["ngon ngu han", "tieng han", "han quoc", "khoa kl"], "CL": ["ngon ngu trung", "tieng trung", "trung quoc", "khoa cl"]
}


class CMCChatbotEngine:
    def __init__(self, artifact_dir: str = "artifacts_model"):
        self.artifact_dir = Path(artifact_dir)

        pipe_path = self.artifact_dir / "dropout_pipeline.joblib"
        meta_path = self.artifact_dir / "model_metadata.json"
        scored_path = self.artifact_dir / "student_snapshots_scored.csv.gz"
        summary_path = self.artifact_dir / "analytics_summary.json"

        for path in (pipe_path, meta_path, scored_path, summary_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Thiếu artifact hoặc file rỗng: {path}"
                )

        manifest_file = self.artifact_dir / "manifest.json"
        if manifest_file.is_file():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            for name in ("dropout_pipeline.joblib", "model_metadata.json", "student_snapshots_scored.csv.gz", "analytics_summary.json", "bao_cao_thuc_nghiem.json"):
                if digest(self.artifact_dir / name) != manifest["files"].get(name):
                    raise ValueError(f"Phiên bản model không khớp mã kiểm tra: {name}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                self.pipeline = joblib.load(pipe_path)

            if not callable(getattr(self.pipeline, "predict_proba", None)):
                raise TypeError("Artifact không hỗ trợ predict_proba")
        except Exception as exc:
            raise RuntimeError(f"Không tải được mô hình: {pipe_path}") from exc

        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.analytics = json.loads(summary_path.read_text(encoding="utf-8"))

        if not isinstance(self.meta, dict) or not self.meta.get("feature_cols"):
            raise ValueError("Metadata mô hình không hợp lệ")

        if not isinstance(self.analytics, dict) or not self.analytics.get("by_cohort"):
            raise ValueError("Tóm tắt thống kê không hợp lệ")

        for key in ("threshold", "test_pr_auc", "test_roc_auc", "test_brier"):
            if key not in self.meta:
                raise ValueError(f"Thiếu trường metadata mô hình: {key}")

        self.df = pd.read_csv(scored_path, low_memory=False,float_precision="round_trip")

        required = {
            "record_uid", "student_id", "snapshot_term_index",
            "class_code", "major_code", "major_name",
            "warning_current", "risk_probability"
        } | set(self.meta["feature_cols"])

        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Thiếu cột snapshot: {sorted(missing)}")

        if self.df.empty:
            raise ValueError("Dữ liệu chấm điểm rỗng")

        if not self.df["risk_probability"].between(0, 1).all():
            raise ValueError("Xác suất đã lưu không hợp lệ")

        self.data_mode = os.environ.get("CMC_DATA_MODE", "DEMO")
        source = Dataset()
        if self.meta.get("dataset_manifest_sha256") and (not manifest_file.is_file() or self.meta["dataset_manifest_sha256"] != digest(source.run / "manifests/generation_manifest.json")):
            raise ValueError("Model và dữ liệu không cùng phiên bản")
        self.ml_df, _ = runtime_view(self.df, source, self.data_mode)
        self.df, self.latest_df = academic_view(self.ml_df, source, self.data_mode)
        self.df = apply_prediction_scope(self.df, self.meta)
        self.latest_df = apply_prediction_scope(self.latest_df, self.meta)
        self.research_revision = source.manifest.get("research_revision")
        self.data_as_of = source.cutoff(self.data_mode).date().isoformat()
        self.df["warning_bool"] = self.df["warning_current"].astype(str).str.lower().eq("true")
        self.latest_df["warning_bool"] = self.latest_df["warning_current"].astype(str).str.lower().eq("true")
        self.active_latest_df = self.latest_df[self.latest_df.state_as_of.eq("ENROLLED")].copy()
        self.active_latest_df['intervention_risk'] = self.active_latest_df.risk_probability.where(self.active_latest_df.prediction_status.eq('VALID'))
        self.df_attempts = visible_attempts(source.table("course_attempts"), self.latest_df)
        catalog = source.table("courses")
        self.course_names = dict(zip(catalog.course_id, catalog.course_name))
        self.course_catalog = catalog.set_index("course_id")
        # Tính lại thống kê theo mốc dữ liệu, không dùng thống kê cũ.
        from core.analytics import build_analytics_summary
        self.analytics = build_analytics_summary(self.df, self.df_attempts, self.meta["threshold"], len(self.ml_df))

    def resolve_course_info(self, course_id: str) -> tuple:
        cid = str(course_id).strip()
        match = re.search(
            r"CURR_\d{4}_([A-Z]{2})_[A-Z]{2}_[A-Za-z]+_C(\d+)",
            cid
        )
        short_code = (
            f"{match.group(1)}-C{int(match.group(2)):03d}"
            if match else cid
        )
        return short_code, self.course_names.get(cid) or cid

    def get_student_records(self, identifier: str) -> pd.DataFrame:
        identifier = identifier.strip().upper()
        return self.df[(self.df['student_id'].astype(str).str.upper() == identifier) | (self.df['record_uid'].astype(str).str.upper() == identifier)].sort_values('snapshot_term_index')

    @staticmethod
    def _optional_bool(value):
        return strict_bool(value)

    def get_student_courses(
        self, record_uid: str, term_filter: int = None,
        only_failed: bool = False
    ) -> list:
        if self.df_attempts is None:
            return []

        sub = self.df_attempts[
            self.df_attempts["record_uid"] == record_uid
        ].copy()

        if term_filter is not None:
            sub = sub[sub["term_index"] == term_filter]
            records = self.get_student_records(record_uid)
            selected = records[records.snapshot_term_index.eq(term_filter)]
            if selected.empty:
                return []
            cutoff = selected.iloc[-1]['snapshot_cutoff']
            sub = sub[sub.available_at.le(cutoff) & sub.score_published_at.le(cutoff)]

        if only_failed:
            sub = sub[
                sub["is_passed"].map(self._optional_bool).eq(False)
            ]

        def optional_number(value, integer=False):
            if pd.isna(value):
                return None
            return int(value) if integer else round(float(value), 2)

        rows = []

        for _, r in sub.sort_values(["term_index", "course_id"]).iterrows():
            cid = str(r["course_id"])
            short_code, course_name = self.resolve_course_info(cid)

            barred = exam_barred(r.get('scheduled_hours'), r.get('unexcused_absent_hours'), r.get('excused_absent_hours'))
            eligible = self._optional_bool(r["exam_eligible"])
            expected = course_grade(r.get('course_final_score10'), r.get('exam_final_score10'),
                                    barred=None if eligible is None else not eligible)
            rows.append({
                "gpa_eligible": (strict_bool(self.course_catalog.loc[cid, "is_gpa_eligible"]) is True and strict_bool(self.course_catalog.loc[cid, "is_certificate_course"]) is False) if hasattr(self,"course_catalog") and cid in self.course_catalog.index else None,
                "attendance_barred": barred,
                "eligibility_review": barred is not None and eligible is not None and barred == eligible,
                "grade_policy_review": expected is not None and expected != r.get('grade_letter'),
                "term": int(r["term_index"]),
                "course_id": cid,
                "short_code": short_code,
                "course_name": course_name,
                "credits": optional_number(r["registered_credits"], True),
                "process_score": optional_number(r["process_score10"]),
                "exam_score": optional_number(r["exam_final_score10"]),
                "final_score": optional_number(r["course_final_score10"]),
                "grade_letter": (
                    None if pd.isna(r["grade_letter"])
                    else str(r["grade_letter"])
                ),
                "is_passed": self._optional_bool(r["is_passed"]),
                "exam_eligible": self._optional_bool(r["exam_eligible"]),
            })

        return rows

    def evaluate_student_warnings(self, row: dict) -> list:
        if row.get('state_as_of') not in (None, 'ENROLLED'):
            return [{"level": "info", "title": "Hồ sơ lịch sử", "detail": "Sinh viên không đang học tại mốc dữ liệu."}]
        alerts = []
        def add(level, title, detail):
            alerts.append(dict(level=level, title=title, detail=detail))
        reasons = warning_reasons(row.get('snapshot_term_index'), row.get('gpa4_term'),
                                  row.get('credits_registered_term'), row.get('credits_failed_term'),
                                  row.get('credits_owed_cumulative'))
        for k, title in [('gpa','GPA học kỳ dưới ngưỡng'),('failed','Trượt quá 50% tín chỉ'),('owed','Nợ quá 24 tín chỉ')]:
            if reasons[k] is True:
                add("danger", title, "Đủ tiêu chí cảnh báo theo Điều 18.1 tại học kỳ đã ghi nhận.")
        if any(v is None for v in reasons.values()):
            add("info", "Thiếu dữ liệu cảnh báo", "Chưa thể kết luận đầy đủ theo Điều 18.1.")
        exit_reasons = withdrawal_reasons(row.get('warning_streak'), row.get('warnings_total_to_date'),
                                          row.get('counted_duration_semesters'), row.get('degree_type'))
        if any(v is True for v in exit_reasons.values()):
            add("danger", "Đủ tiêu chí xem xét buộc thôi học", "Điều 18.2: quá 2 kỳ liên tiếp, quá 3 lần cảnh báo hoặc quá thời gian tối đa; cần quyết định chính thức.")
        elif any(v is None for v in exit_reasons.values()):
            add("info", "Thiếu dữ liệu xét thôi học", "Cần kiểm tra số lần cảnh báo và thời gian được tính.")
        if number(row.get('warning_streak')) == 2:
            add("warning", "Hai lần cảnh báo liên tiếp", "Chưa vượt ngưỡng liên tiếp; kiểm tra các tiêu chí khác.")
        if number(row.get('gpa4_cumulative'), 0, 4) is not None and rounded(row['gpa4_cumulative']) < 2:
            add("warning", "GPA tích lũy dưới 2.0", "Chưa đủ tiêu chí GPA tốt nghiệp (Điều 20).")
        reg = number(row.get('credits_registered_term'))
        lo, hi = number(row.get('registration_min_credits')), number(row.get('registration_max_credits'))
        if None not in (reg, lo, hi) and not lo <= reg <= hi:
            if row.get('registration_assessment') == 'FINAL_REMAINDER_SIMULATION_EXCEPTION':
                add("info", "Khối lượng cuối chương trình", "Ngoại lệ giả định của dữ liệu mô phỏng; không kết luận vi phạm hoặc đã được trường phê duyệt.")
            else:
                add("warning", "Ngoài giới hạn đăng ký mặc định", f"Điều 12.2e: {lo:.0f}–{hi:.0f} TC; đăng ký {reg:.0f} TC. Thiếu căn cứ xác nhận ngoại lệ.")
        att = number(row.get('attendance_rate_term'),0,100)
        if att is not None and att < 85:
            add("warning", "Chuyên cần thấp", "Chỉ xét cấm thi từ số giờ vắng của từng học phần (Điều 25.1a).")
        debt = number(row.get('tuition_debt_end_vnd'))
        if debt is not None and debt > 0:
            add("info", f"Công nợ học phí: {debt:,.0f}đ", "Kiểm tra nghĩa vụ tài chính (Điều 20).")
        risk = number(row.get('risk_probability'),0,1)
        if row.get('prediction_status') == 'STALE':
            add("info", "Dự báo ML đã hết hạn", "Chỉ tham khảo kết quả lịch sử; không dùng để can thiệp hiện tại.")
        elif row.get('prediction_status') == 'VALID' and risk is not None and risk >= self.meta['threshold']:
            add("warning", f"Nguy cơ thôi học dự báo: {risk*100:.1f}%", "Không phải cảnh báo theo quy chế.")
        return alerts or [{"level":"safe","title":"Không phát hiện vi phạm tại học kỳ đã ghi nhận","detail":"Không phải xác nhận đủ điều kiện tốt nghiệp."}]

    def predict_what_if(self, custom_input: dict) -> dict:
        if self.pipeline is None:
            raise RuntimeError("Mô hình dự báo thôi học không khả dụng")

        X_df = model_inputs(pd.DataFrame([custom_input]), self.meta)

        prob = float(self.pipeline.predict_proba(X_df)[:, 1][0])

        if not math.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError("Mô hình trả xác suất không hợp lệ")

        lvl = risk_level(prob, self.meta['threshold'])

        return {
            "risk_probability": round(prob, 4),
            "risk_percent": round(prob * 100, 2),
            "risk_level": lvl
        }

    def calculate_payment_scenario(self, record_uid, payment):
        if not self.meta.get('model_available_as_of'):
            raise ValueError('Cần mô hình nghiên cứu mới đã kiểm định')
        records = self.get_student_records(record_uid)
        if records.empty:
            raise ValueError('Không tìm thấy hồ sơ')
        row = records.iloc[-1]
        if row['state_as_of'] != 'ENROLLED' or row['prediction_status'] != 'VALID':
            raise ValueError('Chỉ mô phỏng khi sinh viên đang học và dự báo còn hiệu lực')
        matches = self.ml_df.loc[self.ml_df.record_uid.eq(record_uid) & self.ml_df.snapshot_term_index.eq(row['prediction_term']) & self.ml_df.snapshot_cutoff.eq(row['prediction_cutoff'])]
        if len(matches) != 1:
            raise ValueError('Không tìm thấy bộ đặc trưng gốc duy nhất')
        snapshot = matches.iloc[0].to_dict()
        base = self.predict_what_if(snapshot)
        changed = self.predict_what_if(payment_scenario(snapshot, payment))
        return {'before':base,'after':changed,'difference_pp':round(changed['risk_percent']-base['risk_percent'],2), 'note':'Giả định thanh toán trước khi chốt kỳ; kết quả không chứng minh tác động nhân quả.'}


    def evaluate_regulation_trap(self, query: str) -> str:
        q = re.sub(r"\s+", " ", strip_accents(query)).replace(",", ".")
        notes = []
        m = re.search(r'\bgpa\s*(?:=|la|dat)?\s*(-?\d+(?:\.\d+)?)', q)
        if m:
            g = number(m.group(1),0,4)
            if g is None:
                notes.append("Điều 17: GPA phải nằm trong [0,4].")
            elif "khoa luan" in q or "do an" in q:
                notes.append(f"Điều 12.3a: GPA {g:.2f} {'đạt' if g >= 2.5 else 'chưa đạt'} ngưỡng 2.50; còn điều kiện riêng của chương trình.")
            elif "canh bao" in q:
                term = re.search(r'\b(?:hoc\s*ky|(?<!dang )ky|hk)\s*(\d+)\b(?!\s*(?:tin chi|tc)\b)', q)
                if 'ky dau' in q:
                    term = 1
                elif term:
                    term = int(term.group(1))
                if isinstance(term,int) and term < 1:
                    notes.append("Điều 18: Học kỳ phải từ 1 trở lên.")
                elif term:
                    bad = warning_reasons(term,g,None,None,None)['gpa']
                    notes.append(f"Điều 18.1a: {'Bị' if bad else 'Không bị'} cảnh báo vì GPA; còn tiêu chí tín chỉ.")
                else:
                    notes.append("Điều 18.1a: cần biết học kỳ đầu hay kỳ tiếp theo để xét GPA.")
            elif "xep loai" in q or "loai gi" in q:
                rank = next(label for n,label in [(3.6,'Xuất sắc'),(3.2,'Giỏi'),(2.5,'Khá'),(2,'Trung bình'),(1,'Yếu'),(0,'Kém')] if g >= n)
                notes.append(f"Điều 17.3: {g:.2f} xếp loại {rank}.")
        m = re.search(r'(?:khoi luong chuan|chuan)\s*(?:la\s*)?(\d+)\s*(?:tin chi|tc)',q)
        if m:
            lo, hi = registration_bounds(int(m.group(1))) if int(m.group(1)) else (None,None)
            notes.append(f"Điều 12.2e: giới hạn {lo}–{hi} TC." if lo is not None else "Điều 12: Khối lượng chuẩn phải lớn hơn 0.")
        m = re.search(r'truot\s*(\d+)\s*(?:/|tren)\s*(\d+)',q)
        if m:
            failed, reg = map(int,m.groups())
            if failed > reg or reg == 0:
                notes.append("Điều 18: Số tín chỉ đăng ký/trượt không hợp lệ.")
            else:
                notes.append("Điều 18.1b: " + ("Bị" if warning_reasons(None,None,reg,failed,None)['failed'] else "Không bị") + " cảnh báo vì tỷ lệ tín chỉ trượt.")
        # Chỉ kết luận từ số lần cảnh báo người hỏi thực sự cung cấp.
        streak = re.search(r'(?<![\d.,-])(-?\d+(?:\.\d+)?)\s*lan\s*(?:bi\s*)?canh bao(?: hoc tap)?\s*lien tiep|canh bao(?: hoc tap)?\s*(-?\d+(?:\.\d+)?)\s*lan\s*lien tiep', q)
        total = re.search(r'(?<![\d.,-])(?:tong(?: cong)?\s*)?(-?\d+(?:\.\d+)?)\s*lan\s*(?:bi\s*)?canh bao(?: hoc tap)?\s*(?:khong lien tiep|khong lien tuc)|canh bao(?: hoc tap)?\s*(-?\d+(?:\.\d+)?)\s*lan\s*(?:khong lien tiep|khong lien tuc)', q)
        if streak or total:
            s = whole(next(v for v in streak.groups() if v is not None)) if streak else None
            t = whole(next(v for v in total.groups() if v is not None)) if total else None
            try:
                if (streak and s is None) or (total and t is None): raise ValueError('Số lần không hợp lệ')
                reasons = withdrawal_reasons(s, t, None, None)
                for key, label in [('consecutive','liên tiếp'),('total','không liên tiếp (tổng số lần)')]:
                    if reasons[key] is not None:
                        notes.append(f"Điều 18.2: {'Đủ' if reasons[key] else 'Chưa đủ'} tiêu chí buộc thôi học theo số lần cảnh báo {label}; còn tiêu chí khác và quyết định của trường.")
            except ValueError:
                notes.append('Điều 18.2: Số lần cảnh báo không nhất quán; cần kiểm tra lại.')
        m = re.search(r'\bno(?: dong)?\s*(-?\d+)\s*(?:tin chi|tc)\b',q)
        if m:
            count = int(m.group(1))
            # Tín chỉ chứng chỉ không thuộc tiêu chí nợ. Không suy đoán phần nợ còn lại.
            rest = q[m.end():]
            excluded = re.match(r'\s*(?:chung chi|mon chung chi|gdqp|gdtc)', rest)
            mixed = any(x in q for x in ('bao gom','gom ca','trong do','ke ca','tinh ca','chung chi','gdqp','gdtc')) and not any(x in q for x in ('khong tinh chung chi','khong gom chung chi','khong ke chung chi','da loai chung chi'))
            component = re.search(r'(?:trong do|bao gom|gom ca|gom)\s*(\d+)\s*(?:tin chi|tc)\s*(?:mon )?chung chi', rest)
            if count<0 or (component and int(component.group(1))>count):
                notes.append('Điều 18.1b: Số tín chỉ nợ hoặc phần chứng chỉ không hợp lệ.')
            elif component:
                counted=count-int(component.group(1))
                notes.append(f"Điều 18.1b: Nợ tính cảnh báo {counted} tín chỉ sau khi loại {int(component.group(1))} tín chỉ chứng chỉ. " + ('Bị' if warning_reasons(None,None,None,None,counted)['owed'] else 'Không bị') + ' cảnh báo theo tiêu chí này; còn tiêu chí khác.')
            elif excluded:
                notes.append("Điều 18.1b: Tín chỉ chứng chỉ đã nêu không tính vào ngưỡng nợ trên 24 tín chỉ. Cần biết số tín chỉ nợ ngoài chứng chỉ và các tiêu chí khác.")
            elif mixed:
                notes.append("Điều 18.1b: Cần tách tín chỉ nợ ngoài chứng chỉ trước khi xét ngưỡng trên 24 tín chỉ; chưa đủ căn cứ kết luận.")
            else:
                notes.append("Điều 18.1b: " + ("Bị" if warning_reasons(None,None,None,None,count)['owed'] else "Không bị") + " cảnh báo do nợ tín chỉ, giả định đã loại môn chứng chỉ; còn tiêu chí khác.")
        u = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:vang\s*)?(?:khong co phep|khong phep)|(?:khong co phep|khong phep)\s*(\d+(?:\.\d+)?)\s*%',q)
        x = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:vang\s*)?co phep|(?<!khong )co phep\s*(\d+(?:\.\d+)?)\s*%',q)
        if u or x:
            un = float(next(v for v in u.groups() if v is not None)) if u else 0
            ex = float(next(v for v in x.groups() if v is not None)) if x else 0
            try:
                barred = exam_barred(100, un, ex)
                notes.append("Điều 25.1a: " + ("Cấm thi và phải học lại." if barred else "Không vượt ngưỡng vắng đã nêu; cần kiểm tra tổng số giờ và lý do khác."))
            except ValueError:
                notes.append("Điều 25: Tỷ lệ vắng không hợp lệ.")
        m = re.search(r'(?:thi cuoi ky|diem thi|thi lai)\s*(?:la|duoc|=)?\s*(-?\d+(?:\.\d+)?)',q)
        if m:
            v = number(m.group(1),0,10)
            if v is None:
                notes.append("Điều 13: Điểm thi phải nằm trong [0,10].")
            elif v <= 2:
                notes.append("Điều 13: điểm liệt, xếp F.")
            elif 'thi lai' in q:
                notes.append("Điều 14.3: điểm học phần sau thi lại tối đa C.")
            else:
                notes.append("Điều 13: Không bị điểm liệt theo điểm thi đã nêu; còn phụ thuộc tổng điểm và điều kiện thi.")
        return "<div class='text-xs'>"+"<br>".join(notes)+"</div>" if notes else ""


    def answer_personal_metric_query(self, query: str, records: pd.DataFrame) -> str:
        q_ascii = strip_accents(query)
        term=self.parse_flexible_query(query)['term_filter']
        if term is not None:
            candidates=records[records.snapshot_term_index.le(term)]
            if candidates.empty or not candidates.snapshot_term_index.eq(term).any():
                return '<p>Chưa có kết quả học kỳ được yêu cầu tại mốc dữ liệu.</p>'
            records=candidates.sort_values('snapshot_term_index')
        latest = row_to_clean_dict(records.iloc[-1])
        prev = row_to_clean_dict(records.iloc[-2]) if len(records) >= 2 else None
        is_delta = any(k in q_ascii for k in ["thay doi the nao", "so voi hoc ky truoc", "so voi ky truoc", "tang hay giam"])

        metric_map = [
            (["gpa hoc ky", "gpa ky gan nhat", "gpa ky nay"], "gpa4_term", "GPA học kỳ (Hệ 4)", "{:.2f}"),
            (["gpa tich luy", "diem trung binh tich luy"], "gpa4_cumulative", "GPA tích lũy (Hệ 4)", "{:.2f}"),
            (["tin chi da dat", "tin chi tich luy", "tich luy bao nhieu tin"], "credits_earned_cumulative", "Số tín chỉ tích lũy đã đạt", "{:.0f} TC"),
            (["tin chi no dong", "no bao nhieu tin"], "credits_owed_cumulative", "Số tín chỉ nợ đọng tích lũy", "{:.0f} TC"),
            (["tin chi truot trong hoc ky", "tin chi truot ky nay"], "credits_failed_term", "Số tín chỉ trượt trong học kỳ", "{:.0f} TC"),
            (["ty le chuyen can", "chuyen can cua em"], "attendance_rate_term", "Tỷ lệ chuyên cần học kỳ", "{:.1f}%"),
            (["vang khong phep"], "unexcused_absence_rate_term", "Tỷ lệ vắng không phép", "{:.1f}%"),
            (["so lan canh bao hoc tap", "canh bao may lan"], "warnings_total_to_date", "Tổng số lần cảnh báo học tập", "{:.0f} lần"),
            (["canh bao lien tiep"], "warning_streak", "Số lần cảnh báo liên tiếp", "{:.0f} lần"),
            (["hoc phi con no", "no hoc phi bao nhieu"], "tuition_debt_end_vnd", "Công nợ học phí cuối kỳ", "{:,.0f} VNĐ"),
            (["so ngay qua han", "qua han may ngay"], "tuition_overdue_days_end", "Số ngày quá hạn học phí", "{:.0f} ngày"),
            (["buoi gap co van", "gap co van may lan"], "advisor_contacts_to_date", "Số buổi gặp Cố vấn học tập", "{:.0f} buổi"),
            (["buoi phu dao", "tham gia bao nhieu buoi phu dao"], "tutoring_sessions_to_date", "Số buổi Phụ đạo đã tham gia", "{:.0f} buổi"),
            (["gio tu hoc"], "study_hours_week", "Số giờ tự học mỗi tuần", "{:.1f} giờ/tuần"),
            (["gio lam them"], "part_time_work_hours_week", "Số giờ làm thêm mỗi tuần", "{:.1f} giờ/tuần"),
            (["dang nhap lms"], "lms_logins_4weeks", "Số lần đăng nhập LMS 4 tuần", "{:.0f} lần")
        ]

        matched_lines = []
        for kws, col, label, fmt in metric_map:
            if any(k in q_ascii for k in kws) and col in latest:
                cur_val = number(latest.get(col), -math.inf)
                if cur_val is None:
                    matched_lines.append(f'{label}: Chưa có dữ liệu')
                    continue
                cur_str = fmt.format(cur_val)
                if is_delta and prev is not None and number(prev.get(col), -math.inf) is not None:
                    prev_val = float(prev[col])
                    diff = cur_val - prev_val
                    trend = "tăng" if diff > 0 else ("giảm" if diff < 0 else "không đổi")
                    matched_lines.append(f"• <b>{label}:</b> Kỳ được hỏi (HK{int(latest['snapshot_term_index'])}) là <b>{cur_str}</b>, so với kỳ trước (HK{int(prev['snapshot_term_index'])}: {fmt.format(prev_val)}) đã <b>{trend} {fmt.format(abs(diff))}</b>.")
                else:
                    matched_lines.append(f"• <b>{label} (HK{int(latest['snapshot_term_index'])}):</b> <b>{cur_str}</b>")

        match = re.search(r'(?:hoc lai|cai thien)\s+(\S+)\s+(?:dat|len)\s+(A\+|A|B\+|B|C\+|C|D\+|D|F)(?:\s|$)', q_ascii, flags=re.IGNORECASE)
        if match or any(k in q_ascii for k in ['kich ban hoc lai','gpa hoc lai']) or ('kich ban cai thien' in q_ascii and 'gpa' not in q_ascii):
            if match:
                try:
                    result = self.calculate_retake(latest['record_uid'], {match.group(1):match.group(2).upper()})
                    matched_lines.append(f"Kịch bản học lại: GPA {result['gpa_before']:.2f} → {result['gpa_after']:.2f}; tín chỉ tính GPA {result['gpa_credits']:.0f} (không cộng trùng).")
                except ValueError as exc:
                    matched_lines.append(str(exc))
            else:
                matched_lines.append('Nhập: Kịch bản học lại MÃ_HỌC_PHẦN đạt B. Dùng mã đầy đủ trong bảng điểm.')
        if any(k in q_ascii for k in ["dieu kien tot nghiep", "dieu kien nao de xet tot nghiep", "chung chi the chat", "chuan tieng anh dau ra chua", "du dieu kien tot nghiep"]):
            conditions = graduation_conditions(latest)
            labels = dict(credits='Tín chỉ',mandatory_courses='Học phần bắt buộc',gpa='GPA',english='Ngoại ngữ (SM B1 / GM B2)',physical_education='GDTC',national_defense='GDQP-AN',financial='Tài chính',program_outcomes='Chuẩn đầu ra/nội dung bắt buộc',discipline_and_criminal_clearance='Kỷ luật/TNHS')
            labels['academic_debt'] = 'Nợ học phần'
            status = lambda v: 'Đạt theo hồ sơ mô phỏng' if v is True else 'Chưa đạt' if v is False else 'Chưa đủ bằng chứng xác nhận'
            if latest.get('graduation_review_date'):
                matched_lines.append(f"Ngày xét tốt nghiệp theo hồ sơ mô phỏng: {latest['graduation_review_date']}")
            matched_lines.append("Điều 19–20 (không phải quyết định tốt nghiệp): " + "; ".join(f"{labels[k]}: {status(v)}" for k,v in conditions.items()))

        if not matched_lines:
            return ""
        return f"""<div class="bg-[#131314] border border-blue-500/40 rounded-2xl p-4 text-xs space-y-2 leading-relaxed">
          <div class="text-sm font-bold text-blue-300">👤 Tra cứu Chỉ số Hồ sơ Cá nhân: {latest['student_id']} (Lớp {latest.get('class_code','')})</div>
          <div class="text-[#e3e3e3]">{"<br>".join(matched_lines)}</div></div>"""

    def calculate_retake(self, record_uid, changes, new_gpa=0, new_credits=0):
        records = self.get_student_records(record_uid)
        if records.empty or records.iloc[-1]['state_as_of'] != 'ENROLLED':
            raise ValueError('Không có hồ sơ đang học phù hợp')
        attempts = self.df_attempts.loc[self.df_attempts.record_uid.eq(record_uid)].merge(
            self.course_catalog[['is_gpa_eligible','is_certificate_course']],left_on='course_id',right_index=True,validate='many_to_one')
        ids = {str(c).lower():c for c in attempts.course_id}
        if not isinstance(changes,dict) or len(changes)>30:
            raise ValueError('Danh sách học lại không hợp lệ')
        changes = {ids.get(k.lower(),k):v for k,v in changes.items()}
        last = records.iloc[-1]
        remaining = number(last.get('new_course_credits_available'))
        required=number(last.get('credits_required')); earned=number(last.get('credits_earned_cumulative')); owed=number(last.get('credits_owed_cumulative'))
        cap=number(last.get('registration_max_credits'))
        if None in (remaining,required,earned,owed,cap):
            raise ValueError('Chưa đủ sổ tín chỉ hoặc giới hạn đăng ký học kỳ')
        remaining=min(remaining,max(0,required-earned-owed))
        if number(new_credits,0,remaining) is None:
            raise ValueError('Tín chỉ môn mới vượt số tín chỉ chưa từng học hoặc không hợp lệ')
        selected = attempts.loc[attempts.course_id.isin(changes)].drop_duplicates('course_id')
        if float(new_credits) + selected.registered_credits.sum() > cap:
            raise ValueError('Tổng tín chỉ giả định vượt giới hạn đăng ký học kỳ')
        return retake_projection(attempts.to_dict('records'),changes,new_gpa,new_credits)

    def calc_next_term_gpa_projection(self, row: dict):
        return projection(row) if row.get('state_as_of') in (None,'ENROLLED') else None

    def answer_risk_and_whatif_query(self, query: str, records: pd.DataFrame) -> str:
        q = strip_accents(query)
        if any(k in q for k in ["kich ban hoc lai","gpa hoc lai"]) or ("kich ban cai thien" in q and "gpa" not in q):
            return ""
        if not any(k in q for k in ["du doan","du bao","ky sau","ki sau","ky toi","tuong lai","nguy co","xac suat","what-if","what if","gia lap","kich ban","cong thuc","muc tieu gpa","can dat bao nhieu","cai thien gpa"]):
            return ""
        latest = row_to_clean_dict(records.iloc[-1])
        if latest.get("state_as_of") not in (None,"ENROLLED"):
            return "<div>Hồ sơ lịch sử; không dự báo can thiệp cho sinh viên không đang học.</div>"
        p = self.calc_next_term_gpa_projection(latest)
        if p is None:
            return "<div>Không đủ dữ liệu GPA hoặc không còn tín chỉ mới để lập kịch bản.</div>"
        risk = number(latest.get("risk_probability"),0,1)
        risk_text = "Chưa có dự báo" if risk is None else risk_display(latest, self.meta['threshold'])
        required = f"{p['req_gpa_next']:.2f}" if p['req_gpa_next'] <= 4 else "Không thể đạt chỉ bằng số tín chỉ mới giả định"
        return f"""<div class="bg-[#131314] rounded-xl p-4 text-xs space-y-2">
          <b>Kịch bản GPA sau HK{latest['snapshot_term_index']} (Điều 15, 17)</b>
          <div>Mốc học vụ {latest.get('snapshot_cutoff','')}. Giả định học {p['next_tc']:.0f} tín chỉ mới tính GPA, không học lại/cải thiện. Đây là phép tính, không phải dự báo đã kiểm định.</div>
          <div>Theo điểm chữ đã lưu trong bộ dữ liệu đang sử dụng. Điểm trọng số hiện tại: {p['gpa_quality_points']:.2f}; tín chỉ tính GPA: {p['gpa_credit_count']:.0f}.</div>
          <div>Kịch bản A: GPA kỳ tới {p['pred_gpa_cur']:.2f} → GPA tích lũy {p['pred_cum_cur']:.2f}.</div>
          <div>Kịch bản B: GPA kỳ tới {p['pred_gpa_imp']:.2f} → GPA tích lũy {p['pred_cum_imp']:.2f}.</div>
          <div>GPA kỳ mới tối thiểu cho mục tiêu {p['target_cum']:.2f}: {required}.</div>
          <div>Nguy cơ thôi học ML: {risk_text}. Không phải cảnh báo học tập theo Điều 18. Kịch bản thanh toán (nếu khả dụng) chỉ là phép thử độ nhạy, không khẳng định tác động.</div>
        </div>"""


    def render_student_rich_html(self, records: pd.DataFrame, role: str) -> str:
        latest = row_to_clean_dict(records.iloc[-1])
        warnings = self.evaluate_student_warnings(latest)
        rows_html, bars_html = "", ""
        for _, r in records.iterrows():
            d = row_to_clean_dict(r)
            hk = int(d.get('snapshot_term_index') or 1)
            gpa_t = float(d.get('gpa4_term') or 0)
            gpa_c = float(d.get('gpa4_cumulative') or 0)
            fail_tc = int(d.get('credits_failed_term') or 0)
            fail_pct = float(d.get('failed_credit_ratio') or 0) * 100
            earn_tc = int(d.get('credits_earned_cumulative') or 0)
            req_tc = int(d.get('credits_required') or 135)
            owed_tc = int(d.get('credits_owed_cumulative') or 0)
            att = float(d.get('attendance_rate_term') or 0)
            debt = float(d.get('tuition_debt_end_vnd') or 0)
            gpa_color = "text-rose-400 font-bold" if gpa_t < 1.0 else ("text-amber-300 font-semibold" if gpa_t < 2.0 else "text-emerald-300 font-semibold")
            rows_html += f"""<tr class="border-b border-[#2c2e30]"><td class="py-2 px-2.5 font-bold text-blue-300">HK{hk}</td><td class="py-2 px-2.5 {gpa_color}">{gpa_t:.2f}</td><td class="py-2 px-2.5 font-semibold">{gpa_c:.2f}</td><td class="py-2 px-2.5">{earn_tc}/{req_tc}</td><td class="py-2 px-2.5 {'text-rose-400 font-semibold' if fail_tc > 0 else 'text-slate-400'}">{fail_tc} TC ({fail_pct:.0f}%)</td><td class="py-2 px-2.5 {'text-amber-300 font-bold' if owed_tc > 0 else 'text-slate-400'}">{owed_tc} TC</td><td class="py-2 px-2.5">{att:.0f}%</td><td class="py-2 px-2.5">{debt:,.0f}đ</td></tr>"""
            h_pct = max(8, min(100, int((gpa_t / 4.0) * 100)))
            bar_color = "bg-rose-500" if gpa_t < 1.0 else ("bg-amber-400" if gpa_t < 2.0 else "bg-emerald-400")
            bars_html += f"""<div class="flex flex-col items-center flex-1"><span class="text-[11px] font-bold mb-1">{gpa_t:.2f}</span><div class="w-8 bg-[#131314] h-20 rounded-lg flex items-end p-0.5 border border-[#333537]"><div class="{bar_color} w-full rounded-md" style="height:{h_pct}%"></div></div><span class="text-[10px] text-[#9aa0a6] mt-1">HK{hk}</span></div>"""

        earn_latest = int(latest.get('credits_earned_cumulative') or 0)
        req_latest = max(1, int(latest.get('credits_required') or 135))
        prog_pct = min(100, round(earn_latest / req_latest * 100, 1))
        pass_pct = max(0, round((1.0 - float(latest.get('failed_credit_ratio') or 0)) * 100, 0))
        att_latest = round(float(latest.get('attendance_rate_term') or 0), 1)
        warn_html = "".join([f"""<div class="flex items-start gap-2 bg-[#131314] border border-[#333537] rounded-xl p-2.5 text-xs"><span>{'🔴' if w['level']=='danger' else ('🟡' if w['level']=='warning' else ('🔵' if w['level']=='info' else '🟢'))}</span><div><b class="text-white">{w['title']}:</b> <span class="text-[#c4c7c5]">{w['detail']}</span></div></div>""" for w in warnings])

        return f"""<div class="space-y-4">
          <div class="flex flex-wrap justify-between items-center border-b border-[#333537] pb-2">
            <div><div class="text-sm font-bold text-white">🎓 Sinh viên: <span class="text-blue-300">{latest.get('student_id','')}</span> • Lớp: <span class="text-emerald-300">{latest.get('class_code','')}</span></div>
            <div class="text-[11px] text-[#9aa0a6]">Ngành: {latest.get('major_name','')} ({latest.get('degree_type','')} - Hệ {latest.get('education_mode','')})</div></div>
            <span class="text-[11px] px-2.5 py-0.5 rounded-full bg-blue-950 text-blue-300 font-semibold">{risk_display(latest, self.meta['threshold'])}</span>
          </div>
          <div class="overflow-x-auto bg-[#131314] border border-[#333537] rounded-2xl p-3">
            <div class="text-xs font-semibold text-blue-300 mb-2">1. Bảng tổng hợp kết quả học tập qua các kỳ</div>
            <table class="w-full text-left text-xs"><thead><tr class="border-b border-[#333537] text-[#9aa0a6]"><th class="py-1.5 px-2.5">Kỳ</th><th class="py-1.5 px-2.5">GPA Kỳ</th><th class="py-1.5 px-2.5">GPA TL</th><th class="py-1.5 px-2.5">TC Tích lũy</th><th class="py-1.5 px-2.5">TC Trượt kỳ</th><th class="py-1.5 px-2.5">TC Nợ đọng</th><th class="py-1.5 px-2.5">Chuyên cần</th><th class="py-1.5 px-2.5">Nợ học phí</th></tr></thead><tbody>{rows_html}</tbody></table>
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div class="bg-[#131314] border border-[#333537] rounded-2xl p-3"><div class="text-xs font-semibold text-emerald-300 mb-2">2. Biểu đồ GPA qua các học kỳ (Thang 4.0)</div><div class="flex items-end justify-around pt-2">{bars_html}</div></div>
            <div class="bg-[#131314] border border-[#333537] rounded-2xl p-3 space-y-2.5 text-xs">
              <div class="font-semibold text-amber-300">Tỷ lệ học tập trọng tâm (Kỳ hiện tại)</div>
              <div><div class="flex justify-between text-[11px] mb-1"><span>Tiến độ tín chỉ tích lũy</span><b>{earn_latest}/{req_latest} TC (${prog_pct}%)</b></div><div class="w-full bg-[#282a2c] h-2 rounded-full overflow-hidden"><div class="bg-blue-500 h-full" style="width:${prog_pct}%"></div></div></div>
              <div><div class="flex justify-between text-[11px] mb-1"><span>Tỷ lệ qua môn trong kỳ</span><b>{pass_pct:.0f}%</b></div><div class="w-full bg-[#282a2c] h-2 rounded-full overflow-hidden"><div class="bg-emerald-500 h-full" style="width:${pass_pct}%"></div></div></div>
              <div><div class="flex justify-between text-[11px] mb-1"><span>Chuyên cần học kỳ (không phải điều kiện thi từng môn)</span><b>{att_latest}%</b></div><div class="w-full bg-[#282a2c] h-2 rounded-full overflow-hidden"><div class="bg-indigo-400 h-full" style="width:${min(100, att_latest)}%"></div></div></div>
            </div>
          </div>
          <div class="space-y-1.5"><div class="text-xs font-semibold text-rose-300">3. Cảnh báo & Lời khuyên ngắn gọn</div><div class="grid grid-cols-1 gap-1.5">{warn_html}</div></div>
        </div>"""

    def render_course_grades_html(self, student_id: str, courses: list, term_filter: int = None, only_failed: bool = False) -> str:
        if self.df_attempts is None:
            return "<div class='text-amber-300'>Chưa có dữ liệu học phần. Không thể hiển thị bảng điểm chi tiết.</div>"
        if not courses:
            return "<div class='text-[#9aa0a6]'>Không tìm thấy học phần phù hợp với phạm vi tra cứu.</div>"
        title = f"Bảng điểm chi tiết học phần {'bị trượt (F)' if only_failed else ''} của {student_id}" + (f" (Học kỳ {term_filter})" if term_filter else " (Tất cả học kỳ)")
        rows = ""
        for c in courses:
            if c.get("eligibility_review"):
                status_badge = "Cần đối chiếu điều kiện thi"
            elif c["exam_eligible"] is False:
                status_badge = "Cấm thi"
            elif c["is_passed"] is None:
                status_badge = "Chưa có kết quả"
            elif c["is_passed"]:
                status_badge = "Đạt"
            else:
                status_badge = "Không đạt"

            if c.get("eligibility_review") or c.get("grade_policy_review"):
                status_badge += " · Cần đối chiếu quy chế"
            c = {
                k: ("—" if v is None else v)
                for k, v in c.items()
            }
            rows += f"""<tr class="border-b border-[#2c2e30]"><td class="py-2 px-2 font-bold text-blue-300 whitespace-nowrap">HK{c['term']}</td><td class="py-2 px-2 font-mono text-[11px] text-[#9aa0a6] whitespace-nowrap" title="{c['course_id']}">{c['short_code']}</td><td class="py-2 px-2 font-medium text-white">{c['course_name']}</td><td class="py-2 px-2 text-center">{c['credits']}</td><td class="py-2 px-2 text-center">{c['process_score']}</td><td class="py-2 px-2 text-center">{c['exam_score']}</td><td class="py-2 px-2 text-center font-bold">{c['final_score']}</td><td class="py-2 px-2 text-center font-bold {'text-rose-400' if c['grade_letter']=='F' else 'text-emerald-300'}">{c['grade_letter']}</td><td class="py-2 px-2 text-right">{status_badge}</td></tr>"""
        return f"""<div class="bg-[#131314] border border-[#333537] rounded-2xl p-3.5 space-y-2.5"><div class="text-xs font-bold text-blue-300">📚 {title}</div><div class="overflow-x-auto max-h-80"><table class="w-full text-left text-xs"><thead><tr class="border-b border-[#333537] text-[#9aa0a6]"><th class="py-1.5 px-2">Kỳ</th><th class="py-1.5 px-2">Mã HP</th><th class="py-1.5 px-2">Tên học phần</th><th class="py-1.5 px-2 text-center">TC</th><th class="py-1.5 px-2 text-center">Điểm QT</th><th class="py-1.5 px-2 text-center">Điểm Thi</th><th class="py-1.5 px-2 text-center">Tổng kết</th><th class="py-1.5 px-2 text-center">Chữ</th><th class="py-1.5 px-2 text-right">Trạng thái</th></tr></thead><tbody>{rows}</tbody></table></div></div>"""


    def get_all_classes_summary(self, major_code: str = "") -> list:
        sub = self.active_latest_df
        if major_code:
            sub = sub[sub['major_code'].astype(str).str.upper() == major_code.strip().upper()]
        grp = sub.groupby(['class_code', 'major_code', 'major_name'], as_index=False).agg(
            n_students=('record_uid', 'count'), gpa_mean=('gpa4_cumulative', 'mean'),
            owed_mean=('credits_owed_cumulative', 'mean'), failed_ratio_mean=('failed_credit_ratio', 'mean'),
            warning_count=('warning_bool', 'sum'),
            valid_predictions=('intervention_risk','count'),
            high_risk_count=('intervention_risk', lambda s: int((s >= self.meta['threshold']).sum()))
        ).sort_values(['high_risk_count', 'warning_count'], ascending=False)
        return [{
            "class_code": str(r['class_code']), "major_code": str(r['major_code']), "major_name": str(r['major_name']),
            "n_students": int(r['n_students']), "gpa_mean": round(float(r['gpa_mean']), 2) if pd.notna(r['gpa_mean']) else None,
            "owed_mean": round(float(r['owed_mean']), 1) if pd.notna(r['owed_mean']) else None, "failed_ratio_mean": round(float(r['failed_ratio_mean']), 3) if pd.notna(r['failed_ratio_mean']) else None,
            "warning_count": int(r['warning_count']), "high_risk_count": int(r['high_risk_count']), "valid_predictions":int(r['valid_predictions']), "coverage_pct":round(100*r['valid_predictions']/r['n_students'],1)
        } for _, r in grp.iterrows()]

    def get_class_detail(self, class_code: str) -> dict:
        c_code = class_code.strip().upper()
        sub = self.active_latest_df[self.active_latest_df['class_code'].astype(str).str.upper() == c_code].sort_values('intervention_risk', ascending=False, na_position='last')
        if sub.empty:
            return {}
        n_stu = int(len(sub))
        warn_cnt = int(sub['warning_bool'].sum())
        high_risk_cnt = int((sub['intervention_risk'] >= self.meta['threshold']).sum())
        gpa_avg = number(sub['gpa4_cumulative'].mean(),0,4)
        owed_avg = number(sub['credits_owed_cumulative'].mean())
        fail_avg = number(sub['failed_credit_ratio'].mean()*100,0,100)
        valid_count = int(sub.intervention_risk.notna().sum())
        summary_text = f"Lớp {c_code} ({sub.iloc[0]['major_name']}) có {n_stu} SV • GPA TB: {f'{gpa_avg:.2f}/4.00' if gpa_avg is not None else 'Chưa có'} • Nợ TB: {f'{owed_avg:.1f} TC' if owed_avg is not None else 'Chưa có'} • Cảnh báo học vụ: {warn_cnt} SV • Cần hỗ trợ sớm: {high_risk_cnt} SV (có dự báo hợp lệ: {valid_count}/{n_stu})."
        students_list = [{
            "student_id": str(s['student_id']), "snapshot_term_index": int(s.get('snapshot_term_index') or 1),
            "gpa4_term": number(s.get('gpa4_term'),0,4), "gpa4_cumulative": number(s.get('gpa4_cumulative'),0,4),
            "credits_earned_cumulative": number(s.get('credits_earned_cumulative')), "credits_owed_cumulative": number(s.get('credits_owed_cumulative')),
            "failed_credit_ratio": number(s.get('failed_credit_ratio')), "warning_streak": number(s.get('warning_streak')),
            "attendance_rate_term": number(s.get('attendance_rate_term')), "tuition_debt_end_vnd": number(s.get('tuition_debt_end_vnd')),
            "risk_probability": number(s.get('risk_probability'),0,1), "risk_level": risk_display(row_to_clean_dict(s), self.meta["threshold"]), "prediction_status": s.get("prediction_status"), "risk_display": risk_display(row_to_clean_dict(s), self.meta["threshold"]),
            "gpa_projection": self.calc_next_term_gpa_projection(row_to_clean_dict(s))
        } for _, s in sub.iterrows()]
        return {
            "class_code": c_code, "major_code": str(sub.iloc[0]['major_code']), "major_name": str(sub.iloc[0]['major_name']),
            "n_students": n_stu, "gpa_mean": gpa_avg, "owed_mean": owed_avg, "fail_pct_mean": fail_avg,
            "warning_count": warn_cnt, "valid_predictions": valid_count, "high_risk_count": high_risk_cnt, "summary_text": summary_text, "students": students_list
        }

    def get_faculties_overview(self) -> list:
        sub = self.active_latest_df
        grp = sub.groupby(['major_code', 'major_name'], as_index=False).agg(
            n_students=('record_uid', 'count'), n_classes=('class_code', 'nunique'),
            gpa_mean=('gpa4_cumulative', 'mean'), owed_mean=('credits_owed_cumulative', 'mean'),
            warning_count=('warning_bool', 'sum'),
            valid_predictions=('intervention_risk','count'),
            high_risk_count=('intervention_risk', lambda s: int((s >= self.meta['threshold']).sum())),
            total_debt_vnd=('tuition_debt_end_vnd', 'sum')
        ).sort_values('n_students', ascending=False)
        return [{
            "major_code": str(r['major_code']), "major_name": str(r['major_name']),
            "n_students": int(r['n_students']), "n_classes": int(r['n_classes']),
            "gpa_mean": round(float(r['gpa_mean']), 2) if pd.notna(r['gpa_mean']) else None, "owed_mean": round(float(r['owed_mean']), 2) if pd.notna(r['owed_mean']) else None,
            "warning_count": int(r['warning_count']), "high_risk_count": int(r['high_risk_count']), "valid_predictions":int(r['valid_predictions']), "coverage_pct":round(100*r['valid_predictions']/r['n_students'],1),
            "high_risk_pct": round(float(r['high_risk_count'] / r['valid_predictions'] * 100), 1) if r['valid_predictions'] else None,
            "total_debt_billion": round(float(r['total_debt_vnd'] or 0) / 1e9, 2)
        } for _, r in grp.iterrows()]

    def get_faculty_detail(self, major_code: str) -> dict:
        m_code = major_code.strip().upper()
        sub = self.active_latest_df[self.active_latest_df['major_code'].astype(str).str.upper() == m_code]
        if sub.empty:
            return {}
        classes = self.get_all_classes_summary(m_code)
        n_stu = int(len(sub))
        warn_cnt = int(sub['warning_bool'].sum())
        hr_cnt = int((sub['intervention_risk'] >= self.meta['threshold']).sum())
        gpa_avg = number(sub['gpa4_cumulative'].mean(),0,4)
        owed_avg = number(sub['credits_owed_cumulative'].mean())
        return {
            "major_code": m_code, "major_name": str(sub.iloc[0]['major_name']),
            "n_students": n_stu, "n_classes": len(classes), "gpa_mean": gpa_avg, "owed_mean": owed_avg,
            "warning_count": warn_cnt, "high_risk_count": hr_cnt, "valid_predictions":int(sub.intervention_risk.notna().sum()),
            "summary_text": f"Khoa {sub.iloc[0]['major_name']} ({m_code}) có {len(classes)} lớp ({n_stu:,} SV) • GPA TB: {f'{gpa_avg:.2f}/4.00' if gpa_avg is not None else 'Chưa có'} • Nợ TB: {f'{owed_avg:.1f} TC' if owed_avg is not None else 'Chưa có'} • Cảnh báo học vụ: {warn_cnt:,} SV • Cần hỗ trợ sớm: {hr_cnt:,} SV; có dự báo {sub.intervention_risk.notna().sum()}/{n_stu} SV.",
            "classes": classes
        }


    def parse_flexible_query(self, query: str) -> dict:
        clean_q = re.sub(r'\s*\[(?:khoa|toàn trường)\s*#\d+\]\s*$', '', str(query), flags=re.IGNORECASE).strip()
        q_norm = normalize_vn(clean_q)
        q_ascii = re.sub(r"\s+", " ", strip_accents(q_norm))
        stu_match = re.search(r'\b(B[A-Z]{2}\d{6}|SIMV31_\d{6})\b', clean_q.upper())
        cls_match = re.search(r'\b([A-Z]{2}\d{2}-\d{2})\b', clean_q.upper())
        maj_match = re.search(r'\b(?:KHOA|NGANH)\s+(IT|BA|CS|SE|AI|GD|KL|CL|EM|NS|LS|IB|MC|MK|PR|DA|GA|CB|EC)\b', q_ascii.upper())
        detected_major = maj_match.group(1) if maj_match else ""
        if not detected_major:
            for m_code, aliases in MAJOR_ALIASES.items():
                if any(a in q_ascii for a in aliases):
                    detected_major = m_code
                    break
        term_match = re.search(r'\b(?:hoc\s*ky|(?<!dang )ky|hk)\s*(\d{1,2})\b(?!\s*(?:tin chi|tc)\b)', q_ascii)
        term_num = int(term_match.group(1)) if term_match else None
        art_matches = re.findall(r'\b(?:dieu|quy\s*che|quy\s*dinh|muc)\s*(?:so\s*)?(\d{1,2})\b', q_ascii)
        art_matches = [a for a in art_matches if 1 <= int(a) <= 27]

        course_ascii_kw = ["hoc phan", "mon hoc", "diem mon", "tung mon", "cac mon", "mon truot", "mon rot", "mon no", "chi tiet mon", "diem f", "hoc lai mon", "c00"]
        personal_ascii_kw = [
            "xem diem", "muon xem diem", "diem so", "bang diem", "diem cua toi", "diem cua minh",
            "diem ca nhan", "ket qua hoc tap", "ho so", "bieu do", "cua em", "cua minh", "cua toi",
            "em co bi", "em da", "em con", "du doan", "du bao", "ki sau", "ky sau", "ky toi", "ki toi",
            "what-if", "what if", "gia lap", "muc tieu gpa", "cong thuc du doan"
        ]
        class_dir_kw = ["xem lop", "danh sach lop", "tra cuu lop", "cac lop", "chon lop", "lop hoc", "tim lop"]

        is_course = any(k in q_ascii for k in course_ascii_kw) and not any(k in q_ascii for k in ["ty le", "khoa", "toan truong", "nhieu nhat"])
        is_personal = bool(stu_match) or any(k in q_ascii for k in personal_ascii_kw)
        is_class_dir = any(k in q_ascii for k in class_dir_kw) and not bool(cls_match)

        return {
            "clean_q": clean_q, "q_norm": q_norm, "q_ascii": q_ascii, "articles": art_matches,
            "student_id": stu_match.group(1) if stu_match else "", "class_code": cls_match.group(1) if cls_match else "",
            "major_code": detected_major, "term_filter": term_num,
            "is_course": is_course, "is_personal": is_personal, "is_class_dir": is_class_dir
        }
