"""Nho bo phan lop chu de cau hoi (số Điều hoặc 'none') trên dữ liệu câu hỏi.

Mục đích minh hoạ bước "train AI" theo hướng data-centric:
  - Nhãn: số Điều mong đợi (1..27) hoặc 'none' cho câu ngoài phạm vi.
  - Dữ liệu: seed người thật (CASES/EXTENDED/REGRESSION + hai bo JSON) và bộ câu
    mo rong do scripts.generate_human_questions sinh ra va da loc boi retrieval.
  - Mô hình: TF-IDF (word + char n-gram) + LogisticRegression; đánh giá 5-fold CV.

Lưu ý quan trọng:
  - Script này KHÔNG thay thế core.intent_router (router quy tắc + phân quyền vẫn
    là đường chạy chính); chỉ đo mức độ tách chủ đề của tập câu hỏi.
  - Model lưu trong data/ (gitignored); chỉ script và báo cáo metrics là tư liệu.
  - Dữ liệu suy diễn, không phải tập đánh giá độc lập/mù.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from scripts.benchmark_rag import CASES, EXTENDED, REGRESSION

ROOT = Path(__file__).resolve().parents[1]
SEED_FILES = [
    ROOT / "tests" / "cases" / "bo_cau_ra_soat_40_de_xuat.json",
    ROOT / "tests" / "cases" / "bo_cau_wave2_de_xuat.json",
    ROOT / "tests" / "cases" / "bo_cau_mo_rong_giong_nguoi.json",
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^\w0-9\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def load_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for q, exp in list(CASES) + list(EXTENDED) + list(REGRESSION):
        rows.append((q, "none" if exp is None else str(exp)))
    for path in SEED_FILES:
        if not path.exists():
            continue
        for q, exp in json.loads(path.read_text(encoding="utf-8-sig")):
            rows.append((q, "none" if exp is None else str(exp)))
    dedup: dict[str, tuple[str, str]] = {}
    for q, label in rows:
        key = normalize(q)
        if key and key not in dedup:
            dedup[key] = (q, label)
    return list(dedup.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-out", type=Path, default=ROOT / "data" / "models" / "intent_classifier.joblib")
    ap.add_argument("--metrics-out", type=Path, default=ROOT / "data" / "kiem_tra_rag" / "intent_classifier_metrics.json")
    a = ap.parse_args()

    rows = load_rows()
    counts_all = Counter(label for _, label in rows)
    # StratifiedKFold cần mỗi lớp ít hơn n_splits mẫu; bỏ lớp quá hiếm khỏi CV,
    # vẫn báo cáo số lượng để không che giấu dữ liệu thưa.
    eligible = [c for c in counts_all.values() if c >= 2]
    n_splits = min(5, min(eligible)) if eligible else 2
    n_splits = max(2, n_splits)
    rows = [(q, label) for q, label in rows if counts_all[label] >= n_splits]
    texts = [normalize(q) for q, _ in rows]
    labels = [label for _, label in rows]
    counts = Counter(labels)
    dropped = {k: v for k, v in counts_all.items() if v < n_splits}

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
    pred = cross_val_predict(pipe, texts, labels, cv=cv)
    accuracy = float(sum(p == y for p, y in zip(pred, labels)) / len(labels))
    report = classification_report(labels, pred, output_dict=True, zero_division=0)

    pipe.fit(texts, labels)
    a.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipe, "labels": sorted(set(labels))}, a.model_out)

    metrics = {
        "so_mau_cv": len(rows),
        "so_mau_goc": sum(counts_all.values()),
        "so_lop": len(counts),
        "lop_bo_khi_cv_thua": dropped,
        "phan_bo_lop": dict(sorted(counts.items(), key=lambda kv: (kv[0] == "none", kv[0]))),
        "n_splits": n_splits,
        "cv_accuracy": round(accuracy, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "model_path": str(a.model_out.relative_to(ROOT)),
        "luu_y": "Phụ trợ đo tách chủ đề; không thay router quy tắc; dữ liệu suy diễn, không phải tập mù.",
    }
    a.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    a.metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("so_mau_goc", "so_mau_cv", "so_lop", "n_splits", "cv_accuracy", "macro_f1", "weighted_f1")}, ensure_ascii=False, indent=2))
    print("Model:", a.model_out)
    print("Metrics:", a.metrics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
