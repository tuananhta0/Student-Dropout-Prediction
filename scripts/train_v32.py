"""Independent V3.2 trainer. Default smoke mode; no TEST files accessed.

Full B5 fit is an explicit opt-in for Phase 2-E; running the default mode
cannot overwrite the legacy V2.2 dropout_model.joblib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.features import B5_FEATURES  # noqa: E402
from src.ml_v32 import V32TrainingLoader, make_b5_pipeline, validate_feature_frame  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Train-only Phase 2-C/D V3.2 trainer")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--data-root", type=Path, default=PROJECT / "data/external/v3_2_main")
    p.add_argument("--report-dir", type=Path, default=PROJECT / "reports/ml_v3_2")
    p.add_argument("--model-dir", type=Path, default=PROJECT / "models/v3_2")
    p.add_argument("--seed", type=int, default=221026)
    p.add_argument("--smoke-rows", type=int, default=1500)
    p.add_argument("--smoke-val-rows", type=int, default=100)
    p.add_argument("--confirm-full-fit", action="store_true", help="Required for full model fit")
    args = p.parse_args()
    if args.mode == "full" and not args.confirm_full_fit:
        p.error("Full model training requires --confirm-full-fit")
    if args.smoke_rows < 300 or args.smoke_val_rows < 1:
        p.error("Smoke rows must be >=300 and val rows >=1")

    root, report_dir, model_dir = args.data_root.resolve(), args.report_dir.resolve(), args.model_dir.resolve()
    if report_dir.is_relative_to(root) or model_dir.is_relative_to(root):
        p.error("Output directories must not be under the source dataset root")
    if model_dir == (PROJECT / "models").resolve():
        p.error("Refusing legacy V2.2 model output directory")
    legacy_files = [PROJECT / "models/dropout_model.joblib",
                    PROJECT / "models/intent_classifier.joblib"]
    if any(not file.is_file() for file in legacy_files):
        raise FileNotFoundError("Missing protected V2.2 baseline artifact")
    legacy_before = {file.name: sha256(file) for file in legacy_files}
    loader = V32TrainingLoader(root)
    t0 = perf_counter()
    train = loader.load("TRAIN_2022_23")
    val = loader.load("VALIDATION_2024")
    validate_feature_frame(val.features, reference=train.features)
    if set(train.keys.record_uid).intersection(val.keys.record_uid):
        raise ValueError("UID overlap between Train and Validation")
    if args.mode == "smoke":
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(train.labels), size=min(args.smoke_rows, len(train.labels)), replace=False))
        X, y = train.features.iloc[idx], train.labels.iloc[idx]
        X_val = val.features.iloc[:args.smoke_val_rows]
    else:
        X, y = train.features, train.labels
        X_val = val.features
    if y.nunique() != 2:
        raise ValueError("Training sample does not contain both labels")
    pipe = make_b5_pipeline(train.features, seed=args.seed)
    pipe.fit(X, y)  # preprocessing + model FIT ON TRAIN ONLY
    scores = pipe.predict_proba(X_val)[:, list(pipe.named_steps["clf"].classes_).index(1)]
    if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("Invalid prediction probabilities")
    versions = {name: version(name) for name in
                ("numpy", "pandas", "scikit-learn", "joblib", "xgboost", "lightgbm")}
    inputs = root / "splits/full_simulation"
    evidence = {
        "status": "PASS", "phase": "2", "scope": "C4,C5,D1-D5",
        "mode": args.mode, "target": "dropout_next_2_terms", "scenario": "V3_2_MAIN",
        "train_split": "TRAIN_2022_23", "validation_split": "VALIDATION_2024",
        "test_accessed": False, "seed": args.seed,
        "model": "LogisticRegression(solver=liblinear,max_iter=350)",
        "features_ordered": B5_FEATURES,
        "preprocessing": "median+StandardScaler numeric; mode+OneHotEncoder(ignore) categorical, fit on Train only",
        "train_rows_fitted": int(len(X)), "train_positive_fitted": int(y.sum()),
        "validation_rows_transformed": int(len(X_val)),
        "probability_min": float(scores.min()), "probability_max": float(scores.max()),
        "input_sha256": {
            str(f.relative_to(root)): sha256(f)
            for f in (inputs / "X_TRAIN_2022_23.csv.gz", inputs / "y_TRAIN_2022_23.csv.gz",
                      inputs / "X_VALIDATION_2024.csv.gz", inputs / "y_VALIDATION_2024.csv.gz")
        },
        "environment": {"python": platform.python_version(), **versions},
        "elapsed_sec": round(perf_counter() - t0, 3),
        "note": "Synthetic data only. No claims for real CMC students.",
    }
    if args.mode == "full":
        if not model_dir.is_relative_to((PROJECT / "models/v3_2").resolve()):
            p.error("Model artifacts must be stored under models/v3_2")
        artifact = model_dir / "b5_reference_pipeline.joblib"
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite an existing model: {artifact}")
        y_val = val.labels
        evidence["validation_metrics_provisional"] = {
            "pr_auc": float(average_precision_score(y_val, scores)),
            "roc_auc": float(roc_auc_score(y_val, scores)),
            "brier": float(brier_score_loss(y_val, scores)),
            "n": int(len(y_val)), "positive": int(y_val.sum()),
            "note": "Provisional B5 only, not a complete Phase 2-E baseline reproduction",
        }
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipe, "feature_columns": list(B5_FEATURES),
                     "model_version": "v3.2-b5-reference-provisional",
                     "target": "dropout_next_2_terms", "source": "V3_2_MAIN_FULL_SIMULATION"}, artifact)
        evidence["model_artifact"] = str(artifact.relative_to(PROJECT))
        evidence["model_sha256"] = sha256(artifact)
    legacy_after = {file.name: sha256(file) for file in legacy_files}
    if legacy_before != legacy_after:
        raise RuntimeError("Protected V2.2 model artifact changed")
    evidence["legacy_v22_models_unchanged"] = True
    evidence["legacy_v22_sha256"] = legacy_after
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / ("training_smoke_CD.json" if args.mode == "smoke" else "b5_training_provisional.json")
    report_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PHASE_2_CD_{args.mode.upper()}_PASS; train_fitted={len(X):,}; val_scored={len(X_val):,}; report={report_path}")


if __name__ == "__main__":
    main()
