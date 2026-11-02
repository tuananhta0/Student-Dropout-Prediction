import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_VERSION = "3.2.0"
EXPECTED_SCENARIO = "V3_2_MAIN"

EXCLUDED = {
    "weekly_state_hidden",
    "synthetic_shock_events",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_table_path(root, name, filename):
    if name.startswith(("X_FULL_SIMULATION_", "y_FULL_SIMULATION_")):
        return root / "splits/full_simulation" / filename

    if name.startswith(("X_ASOF_2026_DEMO_", "y_ASOF_2026_DEMO_")):
        return root / "splits/asof_2026_demo" / filename

    if name.startswith("snapshot_labels_"):
        return root / "labels" / filename

    if name == "prediction_snapshots":
        return root / "features" / filename

    return root / "master" / filename


def verify_file(path, expected_hash):
    if not path.is_file():
        raise FileNotFoundError(f"Missing: {path}")

    actual = sha256_file(path)

    if actual.lower() != expected_hash.lower():
        raise ValueError(
            f"SHA256 mismatch: {path}\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    root = args.data.resolve()
    manifest_path = root / "manifests/generation_manifest.json"

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    if manifest.get("generator_version") != EXPECTED_VERSION:
        raise ValueError("Unexpected generator version")

    if manifest.get("scenario_id") != EXPECTED_SCENARIO:
        raise ValueError("Unexpected scenario")

    if manifest.get("is_synthetic") is not True:
        raise ValueError("Expected synthetic dataset")

    if manifest.get("n_students") != 20000:
        raise ValueError("Unexpected student count")

    print("PASS: Manifest identity and provenance")

    verified_tables = 0
    skipped_tables = 0

    for name, metadata in manifest["table_artifacts"].items():
        filename = metadata["file"]

        if name in EXCLUDED:
            skipped_tables += 1
            continue

        path = resolve_table_path(root, name, filename)
        verify_file(path, metadata["sha256"])
        verified_tables += 1

    print(f"PASS: {verified_tables} table SHA256 checks")
    print(f"SKIPPED: {skipped_tables} audit-only tables")

    verified_docs = 0

    for name, metadata in manifest["documentation_artifacts"].items():
        path = root / "docs" / metadata["file"]
        verify_file(path, metadata["sha256"])
        verified_docs += 1

    print(f"PASS: {verified_docs} documentation SHA256 checks")

    if verified_tables != 32 or skipped_tables != 2 or verified_docs != 2:
        raise ValueError("Unexpected manifest artifact counts")

    if (root / "audit_only_hidden_latents").exists():
        raise ValueError("Audit-only data must not be in runtime directory")

    print("\nV3.2 MANIFEST VERIFICATION PASS")


if __name__ == "__main__":
    main()
