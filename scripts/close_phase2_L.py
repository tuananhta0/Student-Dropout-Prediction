"""Phase 2-L: read-only evidence verification and controlled Git handover.

No model training, no Test scoring or Test label reading. Only --verify writes
new L evidence; --protect-predictions edits a single .gitignore rule;
--stage-safe stages an explicit allowlist; --stamp-handover fills a commit SHA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/ml_v3_2'
MODEL = ROOT / 'models/v3_2/frozen_rf_b_pipeline.joblib'
META = ROOT / 'models/v3_2/model_metadata.json'
LREPORT = REPORTS / 'phase2_L_verification.json'
HANDOVER = ROOT / 'PHASE_2_ML_Training_Handover.md'
PLACEHOLDER = '__PHASE2_IMPLEMENTATION_COMMIT_SHA__'
IGNORE_LINE = '/reports/ml_v3_2/test_predictions_frozen.csv.gz'
BASE_COMMIT = '5c425dce71313e784a77e6b189636cd56f8d2ba6'
TEST_MODULES = [f'scripts.test_phase2_{name}' for name in ('CD', 'F', 'G', 'H', 'I', 'J', 'K')]
EXPECTED_TESTS = 93

REQUIRED_REPORTS = {
    'preflight.md', 'data_contract_verification.json',
    'feature_contract_verification.json', 'training_smoke_CD.json',
    'b5_training_provisional.json', 'baseline_reproduction.json',
    'baseline_comparison.csv', 'baseline_per_term_validation.csv',
    'model_comparison.json', 'model_comparison.md',
    'model_comparison.csv', 'model_comparison_by_term.csv',
    'model_selection_decision.json', 'model_selection_decision.md',
    'frozen_test_metrics.json', 'frozen_test_metrics.md',
    'per_term_metrics.csv', 'test_bootstrap_ci.json',
    'reproducibility.json', 'test_predictions_frozen.csv.gz',
    'subgroup_audit.json', 'subgroup_audit.md',
    'subgroup_metrics.csv', 'hk1_risk_audit.json', 'cohort_shift_audit.csv',
    'artifact_reproducibility_J.json', 'artifact_reproducibility_J.md',
    'inference_contract_K.json',
}
OPTIONAL_REPORTS = {'f_smoke_model_comparison.json', 'g_smoke_selection.json', 'phase2_L_verification.json'}

SAFE_SOURCE = {
    'src/ml_v32.py', 'src/model_policy_v32.py',
    'src/model_bundle_v32.py', 'src/inference_v32.py',
}
SAFE_SCRIPTS = {
    'scripts/check_phase2_B.py', 'scripts/check_phase2_C_contract.py',
    'scripts/train_v32.py', 'scripts/reproduce_phase2_E.py',
    'scripts/compare_phase2_F.py', 'scripts/freeze_phase2_G.py',
    'scripts/evaluate_phase2_H.py', 'scripts/audit_phase2_I.py',
    'scripts/package_phase2_J.py', 'scripts/check_phase2_K.py',
    'scripts/close_phase2_L.py', 'scripts/test_phase2_CD.py',
    'scripts/test_phase2_F.py', 'scripts/test_phase2_G.py',
    'scripts/test_phase2_H.py', 'scripts/test_phase2_I.py',
    'scripts/test_phase2_J.py', 'scripts/test_phase2_K.py',
    'scripts/test_phase2_L.py',
}
SAFE_DOCS = {f'docs/PHASE_2_{name}_RUNBOOK.md' for name in ('CD','E','F','G','H','I','J','K','L')}
SAFE_DOCS.add('docs/MODEL_CARD_V3_2.md')
SAFE_ROOT = {'.gitignore', 'PHASE_2_ML_Training_Handover.md', 'models/v3_2/model_metadata.json'}
SAFE_REPORTS = {f'reports/ml_v3_2/{name}' for name in REQUIRED_REPORTS | OPTIONAL_REPORTS if name != 'test_predictions_frozen.csv.gz'}
SAFE_PATHS = SAFE_SOURCE | SAFE_SCRIPTS | SAFE_DOCS | SAFE_ROOT | SAFE_REPORTS


class GateError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def sha(path):
    p = Path(path)
    require(p.is_file(), f'Missing file: {p}')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def obj(name):
    path = REPORTS / name
    require(path.is_file(), f'Missing A-K report: {path}')
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    require(isinstance(value, dict), f'Invalid JSON report: {name}')
    return value


def git(*args, allowed=(0,)):
    result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True)
    require(result.returncode in allowed,
            f'git {" ".join(args)} failed (rc={result.returncode}): '
            f'{result.stderr.decode("utf-8", "replace").strip()}')
    return result.stdout.decode('utf-8', 'replace').strip()


def ensure_branch():
    branch = git('branch', '--show-current')
    require(branch == 'Dung', f'Refuse to operate on branch {branch!r}; expected Dung')
    git('merge-base', '--is-ancestor', BASE_COMMIT, 'HEAD')
    return git('rev-parse', 'HEAD')


def evidence():
    """Read-only integrity check; never opens Test files or loads model."""
    ensure_branch()
    for item in REQUIRED_REPORTS:
        require((REPORTS/item).is_file(), f'Missing Phase 2 report: {item}')
    for item in SAFE_SOURCE | SAFE_SCRIPTS | SAFE_DOCS:
        require((ROOT/item).is_file(), f'Missing Phase 2 source/document: {item}')
    require(HANDOVER.is_file() and META.is_file(), 'Missing handover or frozen model metadata')
    require((ROOT/'models/v3_2/b5_reference_pipeline.joblib').is_file(),
            'Missing original B5 reference artifact')
    data = obj('data_contract_verification.json')
    feature = obj('feature_contract_verification.json')
    b5 = obj('b5_training_provisional.json')
    e = obj('baseline_reproduction.json')
    f = obj('model_comparison.json')
    g = obj('model_selection_decision.json')
    h = obj('frozen_test_metrics.json')
    i = obj('subgroup_audit.json')
    j = obj('artifact_reproducibility_J.json')
    k = obj('inference_contract_K.json')
    meta = json.loads(META.read_text(encoding='utf-8-sig'))
    require(data.get('status') == 'PASS' and all(data.get('gates',{}).get('B'+str(n)) == 'PASS' for n in range(1,6)),
            'B1-B5 acceptance evidence mismatch')
    require(all(feature.get('C'+str(n)) == 'PASS' for n in (1,2,3)), 'C1-C3 not verified')
    require(b5.get('status') == 'PASS' and b5.get('train_rows_fitted') == 34304
            and b5.get('validation_rows_transformed') == 13981 and b5.get('test_accessed') is False,
            'D/B5 full training evidence mismatch')
    require(e.get('status') == 'PASS' and f.get('status') == 'PASS' and f.get('mode') == 'full',
            'E/F baseline or model comparison evidence missing')
    require(g.get('status') == 'PASS' and g.get('model_id') == 'RF_B'
            and g.get('threshold',{}).get('threshold') == .27 and g.get('test_accessed') is False,
            'G freeze evidence inconsistent')
    require(h.get('status') == 'PASS' and h.get('test_evaluation_once') is True,
            'One-shot H Test evidence missing')
    require(i.get('status') == 'PASS' and i.get('test_rescored') is False,
            'I audit changed Test or did not pass')
    require(j.get('status') == 'PASS' and j.get('model_refitted') is False
            and j.get('test_rescored') is False, 'J reproducibility evidence inconsistent')
    require(k.get('status') == 'PASS' and k.get('no_fit') is True
            and k.get('no_test_split_opened') is True and k.get('no_labels_accessed') is True
            and k.get('api_direct_max_abs_probability_delta', 1) <= 1e-10,
            'K inference evidence inconsistent')
    model_sha = sha(MODEL)
    metadata_sha = sha(META)
    require(model_sha == g.get('model_sha256') == j.get('frozen_model_sha256')
            == k.get('model_sha256') == meta.get('model_sha256'), 'Frozen model SHA drift')
    require(metadata_sha == j.get('metadata_sha256') == k.get('metadata_sha256'),
            'Frozen metadata SHA drift')
    require(meta.get('feature_count') == 58 and meta.get('threshold') == .27
            and meta.get('model_version') == 'v3.2-rf-b-platt-g1', 'Frozen metadata contract drift')
    require(sha(REPORTS/'frozen_test_metrics.json') == j.get('frozen_test_metrics_sha256'),
            'H report edited after packaging')
    require(sha(REPORTS/'model_selection_decision.json') == j.get('model_selection_decision_sha256'),
            'G report edited after packaging')
    require(sha(REPORTS/'subgroup_audit.json') == j.get('subgroup_audit_sha256'),
            'I report edited after packaging')
    require(sha(REPORTS/'model_comparison.json') == g.get('source_f_report_sha256'),
            'F report edited after selection')
    # Legacy V2.2 hashes recorded at D: no need to open the legacy models.
    legacy = b5.get('legacy_v22_sha256',{})
    for name in ('dropout_model.joblib','intent_classifier.joblib'):
        require(sha(ROOT/'models'/name) == legacy.get(name), f'V2.2 artifact changed: {name}')
    return {'model_sha256':model_sha, 'metadata_sha256': metadata_sha,
            'g_sha256':sha(REPORTS/'model_selection_decision.json'),
            'h_sha256':sha(REPORTS/'frozen_test_metrics.json'),
            'i_sha256':sha(REPORTS/'subgroup_audit.json'),
            'k_sha256':sha(REPORTS/'inference_contract_K.json')}


def protect_predictions():
    ensure_branch()
    path = ROOT/'.gitignore'
    content = path.read_text(encoding='utf-8-sig')
    if IGNORE_LINE not in content.splitlines():
        with path.open('a', encoding='utf-8', newline='\n') as f:
            f.write(('' if content.endswith('\n') else '\n') + '\n' + IGNORE_LINE + '\n')
    ignored = git('check-ignore','-v','--no-index','reports/ml_v3_2/test_predictions_frozen.csv.gz')
    require(ignored, 'Test predictions must be Git ignored')
    print('PHASE_2_L_PRIVACY_PASS; test_predictions_frozen.csv.gz ignored; no files staged')


def run_cmd(args, marker=None):
    print('RUN:', ' '.join(args), flush=True)
    res = subprocess.run(args, cwd=ROOT, text=True, encoding='utf-8', errors='replace',
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = res.stdout
    print('\n'.join(output.splitlines()[-14:]), flush=True)
    require(res.returncode == 0, f'Command failed with exit {res.returncode}: {args}')
    if marker:
        require(marker in output, f'Required marker not found: {marker}')
    return output


def verify():
    require(not LREPORT.exists(), 'L report exists; refuse to overwrite acceptance history')
    values = evidence()
    require(IGNORE_LINE in (ROOT/'.gitignore').read_text(encoding='utf-8-sig').splitlines(),
            'Run --protect-predictions to prevent individual Test labels being committed')
    git('diff','--check')
    git('diff','--cached','--check')
    tests_output = run_cmd([sys.executable,'-B','-m','unittest',*TEST_MODULES,'-v'], marker='OK')
    match = re.search(r'Ran (\d+) tests? in ',tests_output)
    require(match is not None and int(match.group(1)) == EXPECTED_TESTS,
            f'Expected {EXPECTED_TESTS} tests, got {match.group(1) if match else "no count"}')
    run_cmd([sys.executable,'-B','-m','scripts.check_v32_edge_cases_e1_e5'],
            marker='V3.2 E1-E5 EDGE CASES PASS')
    # Validate all frozen evidence again after tests. No Test scoring or training.
    require(values == evidence(), 'A-K evidence changed while verifying tests')
    result = {'status':'READY_FOR_COMMIT', 'phase':'2-L',
              'generated_utc':datetime.now(timezone.utc).isoformat(),
              'tests':{'unit_tests':EXPECTED_TESTS,'unit_tests_passed':EXPECTED_TESTS,
                       'phase1_e1_e5':'PASS'}, 'evidence_gates':{letter:'PASS' for letter in 'ABCDEFGHIJK'},
              'test_rescored':False, 'model_refitted':False,
              'frozen_evidence':values,
              'privacy':{'test_predictions_excluded_from_git':True},
              'git_commit_and_push':'PENDING',
              'L4_L5':'PENDING_COMMIT_AND_HANDOVER',
              'note':'Do not claim Phase 2 complete until staged-path audit, SHA handover, commit and push pass.'}
    REPORTS.mkdir(parents=True, exist_ok=True)
    LREPORT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'PHASE_2_L_VERIFY_PASS; tests={EXPECTED_TESTS}; Phase1 E1-E5=PASS; model=UNCHANGED; Test rescored=NO; report={LREPORT}')


def staged_paths():
    raw = subprocess.run(['git','diff','--cached','--name-only','-z'],cwd=ROOT,
                         capture_output=True,check=True).stdout
    return {p.decode('utf-8','replace').replace('\\','/') for p in raw.split(b'\x00') if p}


def stage_check():
    ensure_branch()
    git('diff','--check')
    git('diff','--cached','--check')
    staged = staged_paths()
    require(staged, 'No staged files to review')
    bad = sorted(staged - SAFE_PATHS)
    require(not bad, f'Unsafe/unapproved staged paths: {bad}')
    # Never permit individual-level Test predictions or source/secret files.
    require(all(not p.lower().endswith(('.joblib','.csv.gz','.zip','.env','.pem','.key')) for p in staged),
            'Binary/label/source/secret staged unexpectedly')
    if 'PHASE_2_ML_Training_Handover.md' in staged:
        require(PLACEHOLDER not in HANDOVER.read_text(encoding='utf-8'),
                'Handover cannot be staged with unresolved commit placeholder')
    print(f'PHASE_2_L_STAGE_PASS; staged={len(staged)}; branch=Dung; no model binary/Test row labels/source staged')


def stage_safe():
    ensure_branch()
    require(LREPORT.is_file() and obj('phase2_L_verification.json').get('status') == 'READY_FOR_COMMIT',
            'Run --verify successfully before staging')
    require(IGNORE_LINE in (ROOT/'.gitignore').read_text(encoding='utf-8-sig').splitlines(),
            'Test predictions are not ignored')
    stage = sorted(p for p in SAFE_PATHS if (ROOT/p).is_file() and
                   (p != 'PHASE_2_ML_Training_Handover.md' or PLACEHOLDER not in HANDOVER.read_text(encoding='utf-8')))
    require(stage, 'No safe files found')
    git('add','--',*stage)
    stage_check()


def stamp_handover():
    implementation_sha=ensure_branch()
    require(implementation_sha != BASE_COMMIT, 'Commit implementation first; cannot stamp baseline SHA')
    require(LREPORT.is_file(), 'Missing L verification report')
    require(not staged_paths(), 'Commit staged code/reports before stamping handover')
    require(not git('diff','--name-only'), 'Tracked files have unstaged modifications')
    require(HANDOVER.is_file(), 'Missing handover document')
    content = HANDOVER.read_text(encoding='utf-8')
    require(content.count(PLACEHOLDER) == 1, 'Handover placeholder missing or already stamped')
    updated = content.replace(PLACEHOLDER, implementation_sha)
    HANDOVER.write_text(updated,encoding='utf-8')
    print(f'PHASE_2_L_HANDOVER_STAMPED; implementation_commit={implementation_sha}; commit handover separately')


def post_push_check():
    ensure_branch()
    require(not git('status','--porcelain'), 'Working tree not clean after handover commit')
    current = git('rev-parse','HEAD')
    remote = git('rev-parse','origin/Dung')
    require(current == remote, 'origin/Dung does not match local HEAD; push and check again')
    text = HANDOVER.read_text(encoding='utf-8')
    require(PLACEHOLDER not in text and 'Phase 2 implementation commit SHA' in text,
            'Handover not stamped with actual implementation SHA')
    require('PHASE_2_ML_Training_Handover.md' in git('ls-files'),
            'Handover file has not been committed')
    require('reports/ml_v3_2/test_predictions_frozen.csv.gz' not in git('ls-files'),
            'Individual Test label predictions were committed: FIX REQUIRED')
    print(f'PHASE_2_L_FINAL_PASS; branch=Dung; HEAD={current}; origin/Dung=MATCH; tree=CLEAN; Test labels=NOT TRACKED')
    print('PHASE_2_COMPLETE: A-L mandatory gates PASS, subject to synthetic-only limitations.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    for option in ('preflight','protect-predictions','verify','stage-safe','stage-check','stamp-handover','post-push-check'):
        action.add_argument('--'+option,action='store_true')
    args=parser.parse_args()
    try:
        if args.preflight:
            hashes=evidence()
            print('PHASE_2_L_PREFLIGHT_PASS; A-K evidence+hashes valid; no Test or model loaded; model_sha='+hashes['model_sha256'])
        elif args.protect_predictions: protect_predictions()
        elif args.verify: verify()
        elif args.stage_safe: stage_safe()
        elif args.stage_check: stage_check()
        elif args.stamp_handover: stamp_handover()
        elif args.post_push_check: post_push_check()
    except (GateError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print('PHASE_2_L_FAIL:',exc,file=sys.stderr)
        raise SystemExit(1) from exc

if __name__=='__main__':
    main()
