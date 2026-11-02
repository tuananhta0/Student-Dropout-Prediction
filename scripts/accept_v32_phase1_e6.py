"""Phase 1 E6 acceptance gate for CMC Synthetic Dataset V3.2.

Read-only: reruns existing verifiers, checks the documented original-finance
exception separately, validates the derived ledger, and checks Git hygiene.
Uses the current interpreter/virtualenv. Does not commit, train or change data.

Run from repository root:
    python scripts/accept_v32_phase1_e6.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data/external/v3_2_main'
PROCESSED = REPO / 'data/processed/v3_2'
DOC = REPO / 'docs/DATA_CONTRACT_V3_2.md'


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)
    print('PASS:', message, flush=True)


def run(label: str, args: list[str], expected_rc: int = 0, must_contain: str | None = None):
    print('\n=== ' + label + ' ===', flush=True)
    result = subprocess.run(args, cwd=REPO, text=True, encoding='utf-8',
                            errors='replace', stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    # Only print the tail for large audits; retain full output for gate checks.
    lines = result.stdout.splitlines()
    for line in lines[-15:]:
        print(line, flush=True)
    if result.returncode != expected_rc:
        raise GateError(f'{label}: exit code {result.returncode}, expected {expected_rc}')
    if must_contain is not None and must_contain not in result.stdout:
        raise GateError(f'{label}: required success marker missing: {must_contain}')
    return result.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_finance():
    print('\n=== E6: FINANCE SOURCE EXCEPTION ===', flush=True)
    report_path = PROCESSED / 'finance_transactions_resolution.json'
    derived_path = PROCESSED / 'finance_transactions_resolved.csv.gz'
    source_path = DATA / 'master/finance_transactions.csv.gz'
    require(report_path.is_file() and derived_path.is_file() and source_path.is_file(),
            'Original, derived ledger and resolution report exist')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    manifest = json.loads((DATA / 'manifests/generation_manifest.json').read_text(encoding='utf-8'))
    require(report.get('dataset_version') == '3.2.0'
            and report.get('scenario_id') == 'V3_2_MAIN'
            and report.get('original_rows') == 303375
            and report.get('documented_duplicate_ids') == 241
            and report.get('removed_zero_value_rows') == 241
            and report.get('derived_rows') == 303134
            and report.get('sum_signed_amount_vnd_unchanged') is True,
            'Finance exception report documents exactly 241 removals')
    actual_source = sha256_file(source_path)
    require(actual_source.lower() == report['original_source_sha256'].lower()
            == manifest['table_artifacts']['finance_transactions']['sha256'].lower(),
            'Original finance ledger remains byte-identical to manifest')
    require(sha256_file(derived_path).lower() == report['derived_sha256'].lower(),
            'Derived finance ledger matches recorded SHA-256')
    source = pd.read_csv(source_path, usecols=['transaction_id', 'signed_amount_vnd'],
                         dtype={'transaction_id': 'string'})
    derived = pd.read_csv(derived_path, usecols=['transaction_id', 'signed_amount_vnd'],
                          dtype={'transaction_id': 'string'})
    require(len(source) == 303375 and len(derived) == 303134
            and source['transaction_id'].duplicated().sum() == 241
            and derived['transaction_id'].is_unique
            and not derived['transaction_id'].isna().any(),
            'Derived ledger resolves original 241 ID collisions')
    raw_total = round(float(pd.to_numeric(source['signed_amount_vnd']).sum()), 2)
    derived_total = round(float(pd.to_numeric(derived['signed_amount_vnd']).sum()), 2)
    require(raw_total == derived_total, 'Finance signed total unchanged after derivation')


def verify_document():
    print('\n=== E6: DATA CONTRACT ===', flush=True)
    require(DOC.is_file(), 'Data Contract exists')
    content = DOC.read_text(encoding='utf-8-sig')
    require('**Trạng thái:** Phase 1, nghiệm thu kỹ thuật E6 PASS (26/26); chờ rà soát và commit.' in content,
            'Document records E6 accepted status')
    require('get_student(uid, asof=None)' in content and '16 trường' in content,
            'Document describes updated D4 student profile API')
    require('E1–E5 gồm ba snapshot' in content,
            'Document includes five edge-case checks')
    require('241 `transaction_id` bị lặp' in content,
            'Document retains explicit finance source exception')


def verify_git():
    print('\n=== E6: GIT HYGIENE ===', flush=True)
    for relative in ('data/external/v3_2_main/master/students.csv.gz',
                     'data/processed/v3_2/dataset_catalog.json',
                     'data/processed/v3_2/finance_transactions_resolved.csv.gz',
                     'data/processed/v3_2/finance_transactions_resolution.json'):
        result = subprocess.run(['git', 'check-ignore', '-q', '--no-index', '--', relative],
                                cwd=REPO, capture_output=True)
        require(result.returncode == 0, f'Git ignores {relative}')
    tracked = subprocess.run(['git', 'ls-files', '--', 'data/external/v3_2_main/', 'data/processed/v3_2/'],
                             cwd=REPO, capture_output=True, text=True, check=True)
    require(not tracked.stdout.strip(), 'No tracked V3.2 external/processed data (legacy V2.2 preserved)')
    staged = subprocess.run(['git', 'diff', '--cached', '--name-only'],
                            cwd=REPO, capture_output=True, text=True, check=True)
    require(not staged.stdout.strip(), 'No changes staged unexpectedly')
    status = subprocess.run(['git', 'status', '-sb'], cwd=REPO,
                            capture_output=True, text=True, check=True)
    print(status.stdout.rstrip(), flush=True)
    require(status.stdout.splitlines()[0].startswith('## Dung'),
            'Currently on expected Dung branch')


def main():
    print('CMC V3.2 PHASE 1 E6 ACCEPTANCE', flush=True)
    require(DATA.is_dir() and PROCESSED.is_dir(), 'Data and processed roots exist')
    py = sys.executable
    checks = [
        ('Manifest and checksums', [py, 'scripts/verify_v32_manifest.py', '--data', str(DATA)],
         'V3.2 MANIFEST VERIFICATION PASS'),
        ('Keys, splits and timestamp basics', [py, 'scripts/validate_v32_data.py', '--data', str(DATA)],
         'V3.2 SCHEMA, SPLIT AND TIME CHECKS PASS'),
        ('Core schema', [py, '-m', 'src.schemas', '--data', str(DATA)],
         'V3.2 CORE SCHEMAS PASS'),
        ('Temporal master audit', [py, 'scripts/audit_v32_temporal_integrity.py', '--data', str(DATA)],
         'V3.2 C3 TEMPORAL AUDIT PASS'),
        ('58-feature B5 contract', [py, '-m', 'scripts.check_v32_feature_contract', '--data', str(DATA)],
         'V3.2 FEATURE CONTRACT PASS'),
        ('Dataset provenance', [py, 'scripts/check_v32_provenance.py', '--data', str(DATA)],
         'V3.2 PROVENANCE CONTRACT PASS'),
        ('Data Access B1-B4', [py, '-m', 'src.data_access'],
         'V3.2 DATA ACCESS B4 PASS'),
        ('Data Access B5', [py, '-m', 'scripts.check_v32_data_access_b5'],
         'V3.2 DATA ACCESS B5 PASS'),
        ('Student profile D4', [py, '-m', 'scripts.check_v32_student_profile_d4'],
         'V3.2 STUDENT PROFILE D4 PASS'),
        ('Five E1-E5 cases', [py, '-m', 'scripts.check_v32_edge_cases_e1_e5'],
         'V3.2 E1-E5 EDGE CASES PASS'),
    ]
    try:
        for label, command, marker in checks:
            run(label, command, must_contain=marker)
        source_log = run('C2 master audit (expected source anomaly)',
                         [py, 'scripts/audit_v32_master_integrity.py', '--data', str(DATA)],
                         expected_rc=2, must_contain='C2 MASTER AUDIT: NOT PASSED')
        anomalies = [line.strip() for line in source_log.splitlines()
                     if line.startswith('- ')]
        require(anomalies == ["- finance_transactions: 241 duplicate key occurrences on ('transaction_id',)"],
                'C2 has exactly the documented source anomaly, no additional failures')
        verify_finance()
        verify_document()
        verify_git()
    except (GateError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print('\nPHASE 1 E6: FAIL:', exc, flush=True)
        raise SystemExit(1) from exc
    print('\n' + '=' * 55)
    print('V3.2 PHASE 1 E6 ACCEPTANCE PASS', flush=True)
    print('Source finance exception remains documented; not a clean-source claim.', flush=True)
    print('No training, app refactor, Git commit or push was performed.', flush=True)


if __name__ == '__main__':
    main()
