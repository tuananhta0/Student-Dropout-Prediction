"""Sinh bo cau hoi mo rong dien ghat tu nhien cua sinh vien (offline, khong can LLM).

Quy trinh "huan luyen du lieu" (data-centric, thay vi huan luyen trong so mo hinh):
  1. Seed: cac cau hoi nguoi that da viet trong CASES/EXTENDED/REGRESSION va hai bo
     JSON trong tests/cases/; moi cau gan voi so Die mong doi hoac null (ngoai pham vi).
  2. Bien the phong cach: them lo mo dau (Cho em hoi / Da cho em hoi / Cho minh xin hoi...),
     doi dai tu (em / minh / toi), them tu ket lich su su (a / nhi / giup em voi) va thay
     the cum dong nghia theo tu dien nho.
  3. Template chu de theo tung Dieu, viet san tren co so tieu de TSV quy che, de mo rong
     cac chu de it cau mau.
  4. Loc chat luong bang chinh bo dinh tuyen offline `RegulationSearch`:
     - cau co Die mong doi: phai hien thi dung Die do trong bang chung;
     - cau null: khong duoc tra bang chung nao va intent thuoc (regulation, none, clarify).
  5. Chuan hoa de trung lap, gioi han so cau moi moi chu de, xuat JSON dung schema
     `[[cau hoi, so Die hoac null], ...]` cho `scripts.benchmark_rag --cases`.

Bo sinh ra la du lieu suy dien tu ngu, KHONG phai tap kiem thu doc lap/mu; xem
docs/bo_cau_mo_rong_giong_nguoi.md. Khong ho tro cau hoi chua ho so sinh vien that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CMC_RAG_CLOUD", "0")

from core.retrieval import RegulationSearch  # noqa: E402
from scripts.benchmark_rag import CASES, EXTENDED, REGRESSION  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED_FILES = [
    ROOT / "tests" / "cases" / "bo_cau_ra_soat_40_de_xuat.json",
    ROOT / "tests" / "cases" / "bo_cau_wave2_de_xuat.json",
]

# Lo mo dau da thay trong cau nguoi viet; uu tien nhung lo tu nhien, rut ngan.
OPENERS = [
    "Cho em hỏi ",
    "Dạ cho em hỏi ",
    "Cho mình xin hỏi ",
    "Tôi muốn hỏi ",
    "Dạ em hỏi ",
    "Em thắc mắc là ",
    "Cho em xin hỏi ",
    "Cho mình hỏi ",
]
# Tu ket lich su, chen truoc dau hoi neu cau chua co va chua co tu lich su.
END_PARTICLES = ["ạ", "nhỉ", "nha", "giúp em với"]

# Thay cum dong nghia giua nguoi noi va thuat ngu quy che (dung o che de offline).
# Khong nhung tu thua bi trung lap (vd "liên tiếp" khong nhung trong thay the).
SYNONYMS = [
    (r"\bbuộc thôi học\b", "đuổi học"),
    (r"\bphúc tra\b", "chấm lại điểm"),
    (r"\bhọc cải thiện điểm\b", "học lại để nâng điểm"),
    (r"\blớp cải thiện\b", "học cải thiện điểm"),
    (r"\bchuyển sang trường khác\b", "chuyển trường"),
    (r"\bđổi trường\b", "chuyển trường"),
    (r"\bchuyển ngành\b", "đổi ngành học"),
    (r"\bcảnh báo học tập\b", "nhắc nhở học tập"),
    (r"\bnghỉ học tạm thời\b", "bảo lưu học tập"),
    (r"\bbảo lưu học tập\b", "nghỉ học tạm thời"),
]

# Tu manh bao day la cau hoi day du, khong phai tu khoa ngan.
STRONG_CUES = (
    "bao nhiêu", "nào", "sao", "thế nào", "ra sao", "khi nào", "mấy", "không",
    "phải", "giúp", "xin hỏi", "muốn hỏi", "thắc mắc", "nghĩa là", "ở đâu",
    "thời điểm", "thời hạn", "thủ tục", "điều kiện",
)

# Template theo Die mong doi. Cau dau la phien ban co the mo dau; giu nguyen am dau.
# Moi dong: (so Die, [cau khoi]). Cau null dung cho thong tin van hanh khong co trong PDF.
PATTERNS: dict[int | None, list[str]] = {
    2: [
        "Mục tiêu đào tạo của trường hướng tới những năng lực nào ạ?",
    ],
    3: [
        "Một tín chỉ học được bao nhiêu giờ vậy ạ?",
        "Một tín chỉ tương ứng bao nhiêu giờ tín chỉ lý thuyết ạ?",
        "Học một tín chỉ thì cần bao nhiêu giờ tự học?",
        "Chương trình cử nhân học tối đa mấy học kỳ là phải xong?",
        "Kỹ sư được kéo dài thời gian học tối đa bao nhiêu kỳ?",
        "Học phần tiên quyết nghĩa là gì ạ?",
        "Học phần tự chọn khác học phần bắt buộc ở chỗ nào?",
    ],
    4: [
        "Môn bắt buộc trượt rồi thì phải xử lý sao ạ?",
        "Không đạt học phần bắt buộc có được chọn môn thay thế không?",
    ],
    5: [
        "Một tiết học trên trường dài bao nhiêu phút vậy?",
        "Tiết học đầu tiên trong ngày bắt đầu lúc mấy giờ ạ?",
    ],
    6: [
        "Trách nhiệm sinh viên theo quy chế là gì ạ?",
    ],
    7: [
        "Cố vấn học tập giúp em những việc gì ạ?",
        "Giáo viên chủ nhiệm lớp chịu trách nhiệm gì với sinh viên?",
    ],
    8: [
        "Kế hoạch đào tạo của trường công bố vào thời điểm nào trong năm?",
        "Thời khóa biểu học kỳ mới được thông báo trước bao lâu?",
    ],
    9: [
        "Lớp nhập học khác lớp học phần được phân biệt thế nào ạ?",
        "Lớp nhập học và lớp học phần được tạo ra thế nào ạ?",
    ],
    10: [
        "Quy mô lớp học thường bao nhiêu sinh viên một lớp ạ?",
    ],
    11: [
        "Học trực tuyến chiếm tối đa bao nhiêu phần trăm của chương trình?",
        "Học trực tuyến cần điều kiện phê duyệt học phần nào ạ?",
        "Phần mềm học trực tuyến được dùng cho những học phần nào?",
    ],
    12: [
        "Đăng ký tín chỉ mỗi kỳ tối thiểu và tối đa bao nhiêu ạ?",
        "Đã quá hạn mà tự ý bỏ học phần thì bị gì?",
        "Rút học phần trong tuần đầu học kỳ được không ạ?",
        "Làm khóa luận tốt nghiệp cần GPA tích lũy tối thiểu bao nhiêu?",
        "Đăng ký khóa luận thì mất khoảng mấy tháng hoàn thành?",
    ],
    13: [
        "Điểm cuối kỳ dưới bao nhiêu thì bị coi là điểm liệt ạ?",
        "Điểm chữ C quy đổi bao nhiêu điểm trên thang 4?",
        "Thang điểm A, B, C được quy đổi sang GPA thế nào ạ?",
        "Điểm thành phần trong học phần được tính trọng số ra sao?",
    ],
    14: [
        "Vắng thi cuối kỳ không có lý do thì được bao nhiêu điểm ạ?",
        "Thi lại thì điểm cao nhất được giới hạn ở mức chữ nào?",
    ],
    15: [
        "Môn đã đạt điểm C thì có học lại để nâng điểm không ạ?",
        "Học cải thiện điểm thì lấy lần cao nhất hay lần cuối cùng?",
        "Nghỉ học quá nhiều tiết bị trượt có phải học lại không?",
    ],
    16: [
        "Muốn phúc tra điểm cuối kỳ thì nộp đơn ở đâu và trong hạn nào ạ?",
        "Phúc tra điểm thành phần thì khi nào có kết quả trả về?",
        "Phúc tra có mất lệ phí không, quy định ở đâu ạ?",
    ],
    17: [
        "GPA tích lũy được tính bằng cách nào ạ?",
        "Xếp loại học lực khá, giỏi dựa trên thang điểm mấy?",
        "Công thức tính điểm trung bình tích lũy ra sao nhỉ?",
    ],
    18: [
        "Bị cảnh báo học tập liên tiếp mấy lần thì buộc thôi học ạ?",
        "Nợ bao nhiêu tín chỉ thì rơi vào diện cảnh báo học tập?",
        "Cảnh báo học tập được tính dựa trên những tiêu chí nào?",
        "Bao nhiêu lần nhắc nhở thì trường xét buộc thôi học?",
    ],
    19: [
        "Muốn xét tốt nghiệp cần đạt chứng chỉ tiếng Anh trình độ nào ạ?",
        "Chuẩn ngoại ngữ B1 hay B2 là điều kiện gì của tốt nghiệp?",
    ],
    20: [
        "Muốn được xét tốt nghiệp cần đủ những điều kiện gì ạ?",
        "Chưa đủ điều kiện tốt nghiệp thì có bị hạ hạng bằng tốt nghiệp không?",
    ],
    21: [
        "Em bị ốm dài ngày xin nghỉ hai học kỳ có được duyệt không ạ?",
        "Nghỉ học tạm thời vì nghĩa vụ quân sự thì thủ tục thế nào?",
        "Bảo lưu xong rồi muốn quay lại học thì phải làm gì?",
        "Nghỉ vì tai nạn điều trị mấy tháng có được bảo lưu không?",
        "Tự nguyện xin thôi học thì hồ sơ và quyền lợi thế nào ạ?",
        "Xin nghỉ một năm để chăm người thân có được chấp nhận không?",
    ],
    22: [
        "Học ngành này muốn chuyển sang ngành khác trong trường cần điều kiện gì ạ?",
        "Chuyển sang trường đại học khác thì cần giấy tờ gì?",
        "Đổi từ ngành IT sang BA thì đăng ký ở đâu ạ?",
    ],
    23: [
        "Đăng ký học song ngành chương trình thứ hai vào thời điểm nào ạ?",
        "Học đồng thời hai chương trình đào tạo cần điều kiện gì?",
    ],
    24: [
        "Chương trình trao đổi sinh viên hợp tác áp dụng cho ngành nào?",
    ],
    25: [
        "Vắng học quá 20% số tiết thì có được dự thi không ạ?",
        "Mang tài liệu trái phép vào phòng thi bị xử lý thế nào?",
        "Nhờ người thi hộ hoặc làm hộ bài thi bị kỷ luật gì ạ?",
    ],
    26: [
        "Khóa luận tốt nghiệp được lưu trữ bao lâu ạ?",
    ],
    27: [
        "Quy chế đào tạo này có hiệu lực thi hành từ thời điểm nào ạ?",
    ],
    None: [
        "Học phí một tín chỉ hiện nay là bao nhiêu tiền ạ?",
        "Học phí mỗi học kỳ đóng khoảng bao nhiêu vậy?",
        "Khi nào thì đến hạn đóng học phí học kỳ tới ạ?",
        "Ngày mấy phải đóng học phí vào tháng 12 năm nay?",
        "Trường có miễn giảm học phí cho sinh viên giỏi không?",
        "Sinh viên khó khăn có được hỗ trợ học bổng hay không ạ?",
        "Thư viện trường mở cửa từ mấy giờ đến mấy giờ?",
        "Lịch nghỉ lễ, Tết của trường năm nay như thế nào ạ?",
        "Gửi xe máy trong khu trường sở mất bao nhiêu tiền một ngày?",
        "Trường có câu lạc bộ thể thao hay đội nhóm nào không?",
        "Ký túc xá sinh viên có loại phòng và giá thế nào ạ?",
        "Điểm chuẩn thi tuyển sinh đầu vào của ngành công nghệ thông tin?",
        "Thẻ sinh viên làm lại ở đâu và mất bao lâu để nhận?",
        "Khu học xá có WiFi miễn phí cho sinh viên không ạ?",
        "Quy chế học vụ áp dụng cho năm 2050 đã ban hành chưa ạ?",
    ],
}

# Tu choi tho khong cham hoac khong phai cau hoi tu nhien.
QUESTION_CUES = (
    "hỏi", "cho em", "cho mình", "cho tôi", "nhờ", "giúp", "thắc mắc", "muốn",
    "không", "phải", "nào", "sao", "thế nào", "ra sao", "nhiêu", "bao nhiêu",
    "khi nào", "mấy", "hết", "bao lâu", "ở đâu", "thì", "vậy", "ạ", "nhỉ",
    "điều kiện", "quy định", "quy chế",
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^\w0-9\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _slot(text: str, modulus: int) -> int:
    """Chon vi tri deterministic (md5) thay cho hash() cua Python (ngau nhien theo tien trinh)."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % modulus


def has_cue(text: str) -> bool:
    low = text.lower()
    return any(cue in low for cue in QUESTION_CUES)


def looks_natural(q: str) -> bool:
    if not (14 <= len(q) <= 170):
        return False
    words = normalize(q).split()
    if len(words) < 5:  # loai tu khoa ngan kieu "quy mo lop hoc"
        return False
    low = q.lower()
    if not ("?" in q or any(cue in low for cue in STRONG_CUES)):
        return False
    if not has_cue(q):
        return False
    if re.search(r"(.)\1{4,}", q):  # ky tu lap qua dai
        return False
    if re.search(r"\b(cái|thì|mà|liên tiếp)\b\s+\1\b", q):
        return False
    return True


def full_sentence_seed(q: str) -> bool:
    """Chi bien the cac cau mau la cau hoi day du; bo tu khoa ngan trong CASES."""
    words = normalize(q).split()
    if len(words) < 5:
        return False
    low = q.lower()
    return "?" in q or any(cue in low for cue in STRONG_CUES)


def apply_synonyms(q: str) -> str:
    out = q
    for pattern, repl in SYNONYMS:
        out = re.sub(pattern, repl, out, count=1)
    return out


def add_particle(q: str) -> str:
    if not q.rstrip().endswith("?"):
        return q
    body = q.rstrip()[:-1].rstrip()
    if body.endswith(("ạ", "nhỉ", "nha", "vậy", "giúp em với", "với", "không", "không?")):
        return q
    particle = END_PARTICLES[_slot(normalize(q), 3)]  # uu tien a/nhi/nha, tranh "voi a"
    return body + " " + particle + "?"


def add_opener(q: str) -> str:
    opener = OPENERS[_slot(normalize(q), len(OPENERS))]
    stripped = q.lstrip()
    # Neu cau da bat dau bang dai tu/mo dau thi bo qua de tranh trung loc.
    if stripped.startswith(("Em ", "Mình ", "Tôi ", "Cho ", "Dạ ")):
        return q
    # Chu cai dau sau mo dau nen viet thuong cho nuoc noi tu nhien ("Cho em hỏi môn...");
    # giu viet hoa cua tu viet tat toan chu (GPA, IT...).
    first_token = stripped.split(" ", 1)[0]
    if stripped[:1].isupper() and not (len(first_token) >= 2 and first_token.isupper()):
        stripped = stripped[:1].lower() + stripped[1:]
    return opener + stripped


def variant_pronoun(q: str) -> str:
    mapping = [("Em ", "Mình "), ("em ", "mình "), ("Tôi ", "Em "), ("tôi ", "em ")]
    for a, b in mapping:
        if q.startswith(a) or a in q[:12]:
            return q.replace(a, b, 1)
    return q


def load_seeds() -> list[tuple[str, int | None]]:
    seeds: list[tuple[str, int | None]] = []
    for q, exp in CASES:
        seeds.append((q, exp))
    for q, exp in EXTENDED:
        seeds.append((q, exp))
    for q, exp in REGRESSION:
        seeds.append((q, exp))
    for path in SEED_FILES:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        for row in data:
            seeds.append((row[0], row[1]))
    return seeds


def build_candidates(seeds: list[tuple[str, int | None]]) -> list[tuple[str, int | None]]:
    candidates: list[tuple[str, int | None]] = []
    seen_norm: set[str] = set()

    def add(q: str, exp: int | None) -> None:
        q = re.sub(r"\s+", " ", q).strip()
        key = normalize(q)
        if not key or key in seen_norm:
            return
        if not looks_natural(q):
            return
        seen_norm.add(key)
        candidates.append((q, exp))

    # Template chu de. Cau null (ngoai pham vi) chi mo dau, khong ghep tu/nghia.
    for exp, patterns in PATTERNS.items():
        for p in patterns:
            add(p, exp)
            add(add_opener(p), exp)
            if exp is not None:
                add(add_particle(p), exp)
                add(apply_synonyms(p), exp)

    # Bien the tu seed nguoi that viet: chi voi cau hoi day du, toi da 1 bien the.
    for q, exp in seeds:
        if not full_sentence_seed(q):
            continue
        variants = [add_opener(q), add_particle(q), variant_pronoun(q)]
        added = 0
        for v in variants:
            if added >= 1:
                break
            before = len(candidates)
            add(v, exp)
            if len(candidates) > before:
                added += 1
    return candidates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "tests" / "cases" / "bo_cau_mo_rong_giong_nguoi.json")
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "kiem_tra_rag" / "cau_mo_rong_giong_nguoi_summary.json")
    ap.add_argument("--max-per-article", type=int, default=6)
    ap.add_argument("--max-null", type=int, default=22)
    ap.add_argument("--dry-run", action="store_true", help="Chi sinh ung vien, khong loc boi retrieval.")
    a = ap.parse_args()

    seeds = load_seeds()
    candidates = build_candidates(seeds)

    accepted: list[tuple[str, int | None]] = []
    rejected: list[dict] = []
    per_article: Counter = Counter()
    search = None if a.dry_run else RegulationSearch()

    for q, exp in candidates:
        if exp is None:
            if per_article[None] >= a.max_null:
                continue
        elif per_article[exp] >= a.max_per_article:
            continue
        if a.dry_run:
            accepted.append((q, exp))
            per_article[exp] += 1
            continue
        out = search.answer(q, cloud=False)
        shown = {c["article"] for c in out.get("regulations", [])}
        if exp is None:
            ok = not out.get("regulations") and out.get("intent", "regulation") in ("regulation", "none", "clarify")
        else:
            ok = exp in shown
        if ok:
            accepted.append((q, exp))
            per_article[exp] += 1
        else:
            rejected.append({"cau_hoi": q, "dieu_mong_doi": exp, "dieu_hien_thi": sorted(shown), "intent": out.get("intent")})

    accepted.sort(key=lambda row: (row[1] is None, row[1] if row[1] is not None else 0, row[0]))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(accepted, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    report = {
        "so_seed": len(seeds),
        "so_ung_vien": len(candidates),
        "so_chap_nhan": len(accepted),
        "so_tu_choi_loc": len(rejected),
        "ty_le_chap_nhan": round(len(accepted) / len(candidates), 3) if candidates else None,
        "phan_bo_theo_dieu": {str(k): v for k, v in sorted(per_article.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
        "luu_y": "Du lieu suy dien tu ngu boi script offline; KHONG phai tap kiem thu doc lap/mu. Nguoi cham phai doi chieu PDF truoc khi cong nhan.",
        "tu_choi_mau": rejected[:15],
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("so_seed", "so_ung_vien", "so_chap_nhan", "so_tu_choi_loc", "ty_le_chap_nhan", "phan_bo_theo_dieu")}, ensure_ascii=False, indent=2))
    print("Ghi bo cau:", a.output)
    print("Ghi bao cao loc:", a.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
