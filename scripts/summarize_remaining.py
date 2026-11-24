"""Summarize A-D acceptance evidence without inventing PASSED results.

Run after exporting cloud, ML, HTTP, browser and independent-review reports.
A missing file is NOT_RUN; reports from a different/dirty commit are STALE.
A manual semantic review cannot be treated as an independent blind review by code.
"""
import argparse
import json
import subprocess
from pathlib import Path

DEFAULT_FILES = {
    'gemini_cloud': 'data/kiem_tra_rag/final_cloud.json',
    'semantic_review': 'data/kiem_tra_rag/final_human_review.json',
    'ml_extended': 'data/kiem_tra_ml/final_extended_STRESS.json',
    'main_audit': 'data/kiem_tra_du_lieu/wave2_MAIN/kiem_tra_du_lieu.json',
    'stress_audit': 'data/kiem_tra_du_lieu/wave2_STRESS/kiem_tra_du_lieu.json',
    'runtime_review': 'data/kiem_tra_ban_giao/bao_cao_runtime_public.json',
    'http_live': 'data/kiem_tra_ban_giao/final_http.json',
    'browser_live': 'data/kiem_tra_ban_giao/final_browser.json',
    'policy_review': 'data/doi_chieu_quy_che/review.json',
}


def read_json(path):
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding='utf-8-sig'))


def evaluate(name, obj):
    if obj is None:
        return 'NOT_RUN', 'Chua co bao cao'
    if name == 'gemini_cloud':
        correct = obj.get('cloud_requested') is True and obj.get('so_cau', 0) >= 30 and obj.get('so_cau_gemini', 0) > 0 and obj.get('cloud_quality_gate') is True
        return ('PASS' if correct else 'NOT_PASSED',
                f"LLM {obj.get('so_cau_gemini',0)}/{obj.get('so_cau',0)}; fallback rate {obj.get('ty_le_fallback')}")
    if name == 'semantic_review':
        if obj.get('so_cau', 0) < 30 or not obj.get('nguoi_cham'):
            return 'NOT_PASSED', 'Can cham it nhat 30 cau va ghi ma nguoi cham'
        return 'HUMAN_REVIEWED', f"{obj['so_cau']} cau; chua chung minh nguoi cham doc lap"
    if name == 'ml_extended':
        s = obj.get('stress', {})
        st = s.get('status') if isinstance(s, dict) else None
        return ('PASS' if st == 'PASS' and obj.get('test_overall', {}).get('n', 0) > 0 else 'NOT_PASSED',
                f"STRESS {st}; model {obj.get('model_version')}")
    if name in ('main_audit', 'stress_audit'):
        errors = obj.get('blocking_errors', {})
        if not isinstance(errors, dict) or not errors:
            return 'NOT_PASSED', 'Bao cao khong co blocking_errors'
        n = sum(int(value) for value in errors.values())
        review = obj.get('review_required', {})
        return ('PASS' if n == 0 else 'FAIL',
                f"{len(errors)} structural checks; {n} blocking violations; review_required {review}")
    if name == 'policy_review':
        registration = obj.get('registration', {})
        missing = obj.get('missing_non_direct')
        return ('REVIEW_REQUIRED' if registration or missing else 'PASS',
                f"Credit exceptions={registration}; missing admissions={missing}; do not invent approvals")
    if name == 'runtime_review':
        okay = obj.get('verified_manifest') is True and obj.get('files_checked', 0) >= 24
        return ('PASS' if okay else 'NOT_PASSED',
                f"Manifest={obj.get('verified_manifest')}; files={obj.get('files_checked')}")
    if name == 'http_live':
        entries = obj.get('checks', [])
        passed = {c.get('name') for c in entries if c.get('status') == 'PASS'}
        expected = {'Scenario business gpa', 'Scenario business payment', 'Login student', 'Login advisor', 'Login admin', 'Student cannot access other profile'}
        okay = obj.get('failed') == 0 and expected <= passed and all(c.get('status') == 'PASS' for c in entries)
        return ('PASS' if okay else 'NOT_PASSED', f"{obj.get('passed')} passed, {obj.get('failed')} failed; scenarios required")
    if name == 'browser_live':
        items = obj.get('checks', [])
        combos = {(c.get('role'), c.get('viewport')) for c in items if c.get('failed') == 0 and c.get('passed', 0) >= 5}
        expected = {(role, view) for role in ('student','advisor','admin') for view in ('desktop','mobile')}
        okay = obj.get('failed') == 0 and expected <= combos
        return ('PASS' if okay else 'NOT_PASSED',
                f"{len(combos)}/6 role-viewport journeys passed; {obj.get('failed')} failed")
    raise KeyError(name)


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def summarize(files, expected_head=None):
    results = {}
    for name, path in files.items():
        obj = read_json(path)
        status, detail = evaluate(name, obj)
        origin = (obj or {}).get('provenance', {})
        if obj is not None and expected_head and origin.get('source_commit'):
            if origin['source_commit'] != expected_head or origin.get('source_dirty') is True:
                status = 'STALE'; detail += '; report from another or dirty commit'
        results[name] = {'status': status, 'detail': detail, 'path': str(path)}
    results['fresh_clone_other_machine'] = {'status': 'MANUAL_PENDING', 'detail': 'Requires actual member on another computer'}
    results['visual_human_review'] = {'status': 'MANUAL_PENDING', 'detail': 'Inspect browser visually, not replaceable by Playwright'}
    results['independent_blind_review'] = {'status': 'MANUAL_PENDING', 'detail': 'Another team member must attest independence and keep questions blind'}
    results['credit_exceptions_policy_review'] = {'status': 'MANUAL_PENDING', 'detail': 'Approval evidence for exceptional credit loads must be reviewed'}
    return {'source_commit': expected_head, 'items': results,
            'all_automated_pass': all(results[k]['status'] == 'PASS' for k in DEFAULT_FILES if k not in ('semantic_review','policy_review')),
            'note': 'Automated gates do not certify independent human judgment, real student outcomes, or causal effects.'}


def markdown(report):
    lines = ['# Acceptance report A-D', '', f"Commit: `{report['source_commit'] or 'unknown'}`", '',
             '| Item | Status | Evidence |', '|---|---|---|']
    for name, value in report['items'].items():
        detail = value['detail'].replace('|', '/').replace('\n', ' ')
        lines.append(f"| `{name}` | **{value['status']}** | {detail} |")
    lines.extend(['', 'No manual item is auto-certified.', ''])
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('data/kiem_tra_ban_giao/final_acceptance.json'))
    for name, default in DEFAULT_FILES.items():
        ap.add_argument('--' + name.replace('_','-'), type=Path, default=Path(default))
    args = ap.parse_args()
    paths = {name: getattr(args, name) for name in DEFAULT_FILES}
    result = summarize(paths, git_head())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    args.output.with_suffix('.md').write_text(markdown(result), encoding='utf-8')
    print(json.dumps({k: v['status'] for k,v in result['items'].items()}, ensure_ascii=False, indent=2))
    print('Acceptance report:', args.output, args.output.with_suffix('.md'))
    return 0  # A pending/failed acceptance is reflected in report, not hidden.


if __name__ == '__main__':
    raise SystemExit(main())
