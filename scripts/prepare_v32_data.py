"""Register an extracted V3.2 dataset without copying source data."""

import argparse
import hashlib
import json
from pathlib import Path

DATASET_VERSION = "3.2.0"
SCENARIO_ID = "V3_2_MAIN"

PROFILE_TABLES = {
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
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        required=True
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    source = args.data.resolve()

    manifest_path = (
        source / "manifests" / "generation_manifest.json"
    )

    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    if (
        manifest.get("generator_version") != DATASET_VERSION
        or manifest.get("scenario_id") != SCENARIO_ID
        or manifest.get("is_synthetic") is not True
        or manifest.get("n_students") != 20000
    ):
        raise ValueError("V3.2 manifest identity mismatch")

    if (source / "audit_only_hidden_latents").exists():
        raise ValueError(
            "Audit-only directory must not exist "
            "in the application data root"
        )

    artifacts = manifest["table_artifacts"]
    tables = {}

    allowed_tables = PROFILE_TABLES | {
        "prediction_snapshots"
    }

    for name in sorted(allowed_tables):
        info = artifacts[name]

        if name == "prediction_snapshots":
            folder = "features"
        else:
            folder = "master"

        filename = info["file"]

        if Path(filename).name != filename:
            raise ValueError(
                f"Unsafe filename in manifest: {name}"
            )

        relative = f"{folder}/{filename}"

        if not (source / relative).is_file():
            raise FileNotFoundError(
                source / relative
            )

        tables[name] = {
            "file": relative,
            "rows": int(info["rows"]),
            "sha256": info["sha256"],
        }

    output = (
        repo
        / "data"
        / "processed"
        / "v3_2"
        / "dataset_catalog.json"
    )

    catalog = {
        "dataset_version": DATASET_VERSION,
        "scenario_id": SCENARIO_ID,
        "synthetic": True,
        "source_root": str(source),
        "generation_manifest_sha256": sha256_file(
            manifest_path
        ),
        "generator_source_zip_sha256": manifest.get(
            "source_zip_sha256"
        ),
        "views": {
            "FULL_SIMULATION":
                "training_and_evaluation_only",
            "ASOF_2026_DEMO":
                "historical_demo_with_asof_filtering",
        },
        "demo_as_of": manifest.get("demo_as_of"),
        "serving_requires_timestamp_filters": True,
        "tables": tables,
    }

    content = json.dumps(
        catalog,
        indent=2,
        ensure_ascii=False
    ) + "\n"

    if output.exists():
        existing = output.read_text(
            encoding="utf-8"
        )

        if existing != content:
            raise FileExistsError(
                "Existing catalog has different content: "
                f"{output}"
            )

        print(
            "PASS: Existing catalog matches source "
            "(no overwrite)"
        )
    else:
        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with output.open(
            "x", encoding="utf-8"
        ) as stream:
            stream.write(content)

        print("PASS: Created local V3.2 catalog")

    print(
        f"PASS: {len(tables)} profile/snapshot "
        "source tables registered"
    )
    print(
        "PASS: Audit-only and label tables "
        "excluded from serving catalog"
    )
    print(f"OUTPUT: {output}")
    print("V3.2 PREPARATION PASS")


if __name__ == "__main__":
    main()
