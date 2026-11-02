"""Create a non-destructive, validated V3.2 finance ledger derivative.

Leaves V3.2 source CSV and manifest untouched. Removes only the 241
confirmed duplicate zero-value POST_STUDY_PAYMENT postings emitted by generator.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.data.resolve()
    out = args.out.resolve()
    manifest_file = root / "manifests/generation_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (manifest.get("generator_version") != "3.2.0"
            or manifest.get("scenario_id") != "V3_2_MAIN"
            or manifest.get("is_synthetic") is not True):
        raise ValueError("Unexpected V3.2 source manifest")

    info = manifest["table_artifacts"]["finance_transactions"]
    source = root / "master" / "finance_transactions.csv.gz"
    if info["file"] != source.name:
        raise ValueError("Unexpected source filename")
    source_sha = sha256_file(source)
    if source_sha.lower() != info["sha256"].lower():
        raise ValueError("Finance source SHA-256 does not match manifest")

    frame = pd.read_csv(source, low_memory=False, dtype={"transaction_id": "string"})
    if len(frame) != 303375 or len(frame) != int(info["rows"]):
        raise ValueError("Unexpected number of financial transactions")
    if frame["transaction_id"].isna().any():
        raise ValueError("Null transaction IDs")

    duplicated = frame.duplicated("transaction_id", keep=False)
    collisions = frame.loc[duplicated]
    groups = collisions.groupby("transaction_id", sort=False)
    if len(groups) != 241 or len(collisions) != 482:
        raise ValueError("Duplicate count differs from documented source exception")
    if not groups.size().eq(2).all():
        raise ValueError("Unexpected number of postings per duplicate ID")
    if not collisions["transaction_id"].str.endswith("_F03").all():
        raise ValueError("Unexpected duplicate transaction ID pattern")
    if not collisions["transaction_type"].eq("POST_STUDY_PAYMENT").all():
        raise ValueError("Unexpected duplicate transaction type")
    if not (pd.to_datetime(collisions["event_at"]) > pd.Timestamp("2026-10-01")).all():
        raise ValueError("Duplicate transaction before demo cutoff")

    amounts = pd.to_numeric(collisions["signed_amount_vnd"], errors="raise")
    collisions = collisions.assign(_amount=amounts)
    counts = collisions.groupby("transaction_id")["_amount"].agg(
        zero=lambda s: int(s.eq(0).sum()),
        negative=lambda s: int(s.lt(0).sum()),
    )
    if not (counts["zero"].eq(1) & counts["negative"].eq(1)).all():
        raise ValueError("Duplicate pairs are not one zero and one negative posting")

    all_other_fields = [c for c in frame if c not in ("transaction_id", "signed_amount_vnd")]
    if not groups[all_other_fields].nunique(dropna=False).eq(1).all().all():
        raise ValueError("Duplicate pairs differ in unexpected metadata fields")

    zero_in_collisions = duplicated & pd.to_numeric(frame["signed_amount_vnd"]).eq(0)
    if int(zero_in_collisions.sum()) != 241:
        raise ValueError("Unexpected number of redundant zero postings")
    cleaned = frame.loc[~zero_in_collisions].copy()
    if len(cleaned) != 303134 or not cleaned["transaction_id"].is_unique:
        raise ValueError("Failed to restore transaction ID uniqueness")

    original_total = round(float(pd.to_numeric(frame["signed_amount_vnd"]).sum()), 2)
    cleaned_total = round(float(pd.to_numeric(cleaned["signed_amount_vnd"]).sum()), 2)
    if original_total != cleaned_total:
        raise ValueError("Financial total changed")

    target = out / "finance_transactions_resolved.csv.gz"
    report = out / "finance_transactions_resolution.json"
    if target.exists() or report.exists():
        raise FileExistsError("Output already exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(
        target,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    metadata = {
        "dataset_version": "3.2.0",
        "scenario_id": "V3_2_MAIN",
        "original_source": "master/finance_transactions.csv.gz",
        "original_source_sha256": source_sha,
        "issue": "Generator reuses _F03 when adding redundant zero-value POST_STUDY_PAYMENT",
        "original_rows": len(frame),
        "documented_duplicate_ids": len(groups),
        "removed_zero_value_rows": int(zero_in_collisions.sum()),
        "derived_rows": len(cleaned),
        "sum_signed_amount_vnd_unchanged": True,
        "derived_sha256": sha256_file(target),
        "policy": "Original source immutable; derived ledger is local and must not replace manifest data",
    }
    report.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("PASS: Source finance SHA-256 matches original V3.2 manifest")
    print("PASS: Exactly 241 documented duplicate IDs, each with one redundant zero posting")
    print("PASS: Duplicate cases occur only after 2026-10-01")
    print("PASS: Derived ledger: 303,134 rows, unique transaction_id, unchanged total")
    print("PASS: Original source and manifest untouched")
    print(f"OUTPUT: {target}")
    print(f"REPORT: {report}")
    print("C2 FINANCE SOURCE EXCEPTION RESOLVED IN DERIVED DATA")


if __name__ == "__main__":
    main()
