"""Validate the V3.2 serving catalog against source provenance."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_TABLES = {
    "students",
    "term_academics",
    "course_attempts",
    "engagement_weekly",
    "finance_terms",
    "student_events",
    "enrollment_terms",
    "state_transitions",
    "support_events",
    "term_calendar",
    "prediction_snapshots",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError("FAIL: " + message)
    print("PASS: " + message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "data/processed/v3_2/dataset_catalog.json"
        ),
    )
    args = parser.parse_args()

    root = args.data.resolve()
    manifest_path = (
        root / "manifests/generation_manifest.json"
    )
    catalog_path = args.catalog.resolve()

    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    with catalog_path.open(encoding="utf-8") as stream:
        catalog = json.load(stream)

    check(
        manifest["generator_version"] == "3.2.0"
        and manifest["scenario_id"] == "V3_2_MAIN"
        and manifest["is_synthetic"] is True
        and manifest["n_students"] == 20000,
        "Source manifest identity",
    )

    check(
        catalog["dataset_version"] == "3.2.0"
        and catalog["scenario_id"] == "V3_2_MAIN"
        and catalog["synthetic"] is True
        and Path(catalog["source_root"]).resolve() == root,
        "Catalog identity and source path",
    )

    check(
        sha256_file(manifest_path)
        == catalog["generation_manifest_sha256"],
        "Manifest SHA-256 matches catalog",
    )

    check(
        catalog["generator_source_zip_sha256"]
        == manifest["source_zip_sha256"],
        "Generator input ZIP provenance preserved",
    )

    check(
        manifest["demo_as_of"] == "2026-10-01"
        and catalog["demo_as_of"] == "2026-10-01",
        "Demo observation date",
    )

    check(
        catalog["views"] == {
            "FULL_SIMULATION":
                "training_and_evaluation_only",
            "ASOF_2026_DEMO":
                "historical_demo_with_asof_filtering",
        },
        "Observation view contract",
    )

    check(
        catalog["serving_requires_timestamp_filters"] is True,
        "Timestamp filters required",
    )

    tables = catalog["tables"]

    check(
        set(tables) == EXPECTED_TABLES,
        "Exactly 11 approved tables registered",
    )

    artifacts = manifest["table_artifacts"]

    for name, info in tables.items():
        expected = artifacts[name]

        folder = (
            "features"
            if name == "prediction_snapshots"
            else "master"
        )

        expected_path = (
            f"{folder}/{expected['file']}"
        )

        check(
            info["file"] == expected_path
            and int(info["rows"]) == int(expected["rows"])
            and info["sha256"] == expected["sha256"],
            f"{name}: metadata matches manifest",
        )

        path = (root / info["file"]).resolve()

        if (
            not path.is_relative_to(root)
            or not path.is_file()
        ):
            raise ValueError(
                f"Unsafe or missing source: {name}"
            )

    check(
        not any(
            name.startswith(("snapshot_labels_", "X_", "y_"))
            or "hidden" in name
            or "finance_transactions" == name
            for name in tables
        ),
        "Labels, hidden data and raw finance transactions excluded",
    )

    print("\nV3.2 PROVENANCE CONTRACT PASS")


if __name__ == "__main__":
    main()
