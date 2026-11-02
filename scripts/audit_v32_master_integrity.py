"""Non-destructive primary-key and foreign-key audit of V3.2 master tables."""
import argparse
import json
from pathlib import Path
import pandas as pd

KEYS = {
    "students": ("record_uid",),
    "term_academics": ("record_uid", "term_index"),
    "course_attempts": ("attempt_id",),
    "engagement_weekly": ("record_uid", "term_index", "teaching_week"),
    "finance_terms": ("record_uid", "term_index"),
    "student_events": ("event_id",),
    "enrollment_terms": ("record_uid", "term_index"),
    "state_transitions": ("record_uid", "event_id"),
    "support_events": ("support_event_id",),
    "term_calendar": ("calendar_term_id",),
    "academic_requirement_events": ("record_uid", "term_index", "requirement_type"),
    "course_assessment_components": ("attempt_id", "component_type"),
    "courses": ("course_id",),
    "curricula": ("curriculum_id",),
    "curriculum_courses": ("curriculum_id", "course_id"),
    "finance_transactions": ("transaction_id",),
    "graduation_clearance": ("record_uid", "term_index"),
}

FK_FIELDS = {
    "students": ("curriculum_id",),
    "term_academics": ("record_uid", "calendar_term_id"),
    "course_attempts": ("record_uid", "calendar_term_id", "course_id"),
    "engagement_weekly": ("record_uid", "calendar_term_id"),
    "finance_terms": ("record_uid", "calendar_term_id"),
    "student_events": ("record_uid",),
    "enrollment_terms": ("record_uid", "calendar_term_id"),
    "state_transitions": ("record_uid",),
    "support_events": ("record_uid",),
    "academic_requirement_events": ("record_uid",),
    "course_assessment_components": ("record_uid", "attempt_id"),
    "curriculum_courses": ("curriculum_id", "course_id"),
    "finance_transactions": ("record_uid",),
    "graduation_clearance": ("record_uid",),
}
REF_TABLE = {
    "record_uid": ("students", "record_uid"),
    "calendar_term_id": ("term_calendar", "calendar_term_id"),
    "course_id": ("courses", "course_id"),
    "curriculum_id": ("curricula", "curriculum_id"),
    "attempt_id": ("course_attempts", "attempt_id"),
}


def main(root):
    manifest = json.loads((root / "manifests/generation_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("generator_version") != "3.2.0" or manifest.get("scenario_id") != "V3_2_MAIN":
        raise ValueError("Wrong V3.2 manifest")

    def source(name):
        info = manifest["table_artifacts"][name]
        path = root / "master" / info["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    references = {}
    for col, (name, key) in REF_TABLE.items():
        s = pd.read_csv(source(name), usecols=[key], dtype={key: "string"})[key]
        if s.isna().any():
            raise ValueError(f"Reference {name}.{key} has null values")
        references[col] = set(s)

    errors = []
    print("=== MASTER PRIMARY KEYS AND FOREIGN KEYS ===", flush=True)

    for name, key in KEYS.items():
        fields = sorted(set(key) | set(FK_FIELDS.get(name, ())))
        if name == "finance_transactions":
            fields += ["signed_amount_vnd", "transaction_type"]
        dtype = {col: "string" for col in fields if col in REF_TABLE or col in key and col.endswith("_id")}
        frame = pd.read_csv(source(name), usecols=fields, dtype=dtype, low_memory=False)
        expected = int(manifest["table_artifacts"][name]["rows"])
        if len(frame) != expected:
            errors.append(f"{name}: expected {expected}, found {len(frame)}")

        nulls = int(frame[list(key)].isna().any(axis=1).sum())
        dupes = int(frame.duplicated(list(key)).sum())
        missing = {}
        for col in FK_FIELDS.get(name, ()):
            missing[col] = int((~frame[col].isin(references[col])).sum())
        if nulls or any(missing.values()):
            errors.append(f"{name}: key_nulls={nulls}, missing_FK={missing}")
        if dupes:
            errors.append(f"{name}: {dupes} duplicate key occurrences on {key}")
        marker = "PASS" if not (nulls or dupes or any(missing.values()) or len(frame) != expected) else "ISSUE"
        print(f"{marker}: {name}: {len(frame):,} rows; duplicate keys={dupes}; null keys={nulls}; missing FKs={sum(missing.values())}", flush=True)
        if name == "finance_transactions" and dupes:
            group = frame.loc[frame.duplicated(list(key), keep=False)]
            print(f"  DETAIL: {group.transaction_id.nunique()} reused IDs across {len(group)} rows", flush=True)
            print(f"  Zero-amount records among collisions: {int(group.signed_amount_vnd.eq(0).sum())}", flush=True)
            print(f"  Example transaction IDs: {group.transaction_id.drop_duplicates().head(3).tolist()}", flush=True)

    if errors:
        print("\nC2 MASTER AUDIT: NOT PASSED (source anomalies detected)", flush=True)
        for error in errors:
            print("-", error, flush=True)
        raise SystemExit(2)
    print("\nC2 MASTER AUDIT PASS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    main(args.data.resolve())
