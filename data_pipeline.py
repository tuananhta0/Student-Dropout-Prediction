"""V3.2.1 source contract and point-in-time views. No model training here."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ['record_uid', 'snapshot_term_index']
ROOT = Path(__file__).resolve().parent
MAIN = Path('artifacts/runs/V3_2_MAIN')
MODES = {'ASOF_2026_DEMO': 'demo_as_of', 'FULL_SIMULATION': 'simulation_as_of'}
EXIT_TYPES = {'VOLUNTARY_WITHDRAWAL', 'FORCED_ACADEMIC_WITHDRAWAL',
              'CONFIRMED_INSTITUTIONAL_ABANDONMENT', 'DISCIPLINARY_WITHDRAWAL'}
RUNTIME_TERM = ['part_time_work_hours_week', 'study_hours_week',
                'credits_attempted_cumulative', 'courses_failed_cumulative',
                'retake_count_cumulative', 'registration_rule_status',
                'registration_min_credits', 'registration_max_credits']
EXTRA_FEATURES = ['ca_score10_min', 'ca_exam10_min', 'ca_ineligible_exams', 'ca_unexcused_hours_sum']
TABLES = {
    'students': ('master/students.csv.gz', ['record_uid', 'student_id', 'scenario_id', 'cohort_year', 'curriculum_id', 'admission_method', 'admission_score', 'admission_score_normalized', 'admission_score_missing_reason']),
    'courses': ('master/courses.csv.gz', ['course_id', 'course_name', 'is_gpa_eligible', 'is_certificate_course']),
    'curricula': ('master/curricula.csv.gz', ['curriculum_id', 'degree_type', 'credits_required', 'designed_semesters', 'max_semesters']),
    'curriculum_courses': ('master/curriculum_courses.csv.gz', ['curriculum_id', 'course_id']),
    'term_calendar': ('master/term_calendar.csv.gz', ['calendar_term_id', 'cohort_year', 'term_index', 'grade_release_date']),
    'enrollment_terms': ('master/enrollment_terms.csv.gz', ['record_uid', 'term_index', 'calendar_term_id', 'available_at']),
    'course_attempts': ('master/course_attempts.csv.gz', ['attempt_id', 'record_uid', 'term_index', 'calendar_term_id', 'course_id', 'attempt_no', 'registered_credits', 'process_score10', 'exam_final_score10', 'course_final_score10', 'grade_letter', 'grade_point4', 'is_passed', 'exam_eligible', 'scheduled_hours', 'unexcused_absent_hours', 'excused_absent_hours', 'available_at', 'score_published_at']),
    'term_academics': ('master/term_academics.csv.gz', ['record_uid', 'term_index', 'calendar_term_id', 'available_at', 'snapshot_cutoff', 'credits_registered_term', 'credits_passed_term', 'credits_failed_term', 'credits_earned_cumulative', 'credits_owed_cumulative', 'gpa4_term', 'gpa4_cumulative', 'warning_current', *RUNTIME_TERM]),
    'academic_requirement_events': ('master/academic_requirement_events.csv.gz', ['record_uid', 'requirement_type', 'completion_status', 'available_at']),
    'state_transitions': ('master/state_transitions.csv.gz', ['record_uid', 'event_id', 'from_status', 'to_status', 'effective_at']),
    'student_events': ('master/student_events.csv.gz', ['record_uid', 'event_id', 'event_type', 'term_index', 'effective_at', 'available_at']),
    'prediction_snapshots': ('features/prediction_snapshots.csv.gz', [*KEY, 'snapshot_cutoff', 'feature_source_latest_at', 'student_id', 'cohort_year']),
    'snapshot_labels_FULL_SIMULATION': ('labels/snapshot_labels_full_simulation.csv.gz', [*KEY, 'snapshot_cutoff', 'scenario_id', 'observation_view_id', 'dropout_next_2_terms', 'label_observed', 'label_observation_end', 'observation_as_of', 'split', 'first_dropout_event_id']),
    'snapshot_labels_ASOF_2026_DEMO': ('labels/snapshot_labels_asof_2026_demo.csv.gz', [*KEY, 'snapshot_cutoff', 'scenario_id', 'observation_view_id', 'dropout_next_2_terms', 'label_observed', 'label_observation_end', 'observation_as_of', 'split', 'first_dropout_event_id']),
    'feature_registry': ('docs/feature_registry.csv', ['feature_name', 'role']),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique(df, keys, name):
    require(not df[keys].isna().any().any() and not df.duplicated(keys).any(),
            f'{name}: null or duplicate key {keys}')


def dates(values):
    result = pd.to_datetime(values, format='%Y-%m-%d', errors='raise')
    require(result.notna().all(), 'Missing required timestamp')
    return result


def main_run():
    return Path(os.environ.get('CMC_DATASET_ROOT', ROOT / 'dataset_full')).resolve() / MAIN


class Dataset:
    def __init__(self, run=None):
        self.run = Path(run) if run is not None else main_run()
        manifest = self.run / 'manifests/generation_manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError(f'MAIN dataset missing: {manifest}; run python install_dataset.py <archive>')
        self.manifest = json.loads(manifest.read_text(encoding='utf-8'))
        require(self.manifest.get('scenario_id') == 'V3_2_MAIN' and
                self.manifest.get('identity_migration_version') == '3.2.1' and
                self.manifest.get('n_students') == 20000 and self.manifest.get('is_synthetic') is True,
                'Expected synthetic V3.2.1 MAIN (not BASE/STRESS)')
        self.cache = {}

    def table(self, name):
        if name not in self.cache:
            rel, columns = TABLES[name]
            path = self.run / rel
            contract = self.manifest['documentation_artifacts' if name == 'feature_registry' else 'table_artifacts'][name]
            require(path.is_file() and digest(path) == contract['sha256'], f'Invalid source hash: {rel}')
            df = pd.read_csv(path, low_memory=False)
            require(set(columns) <= set(df.columns), f'{rel}: missing {set(columns)-set(df.columns)}')
            require(len(df) == contract['rows'] and len(df.columns) == contract['columns'], f'{rel}: shape mismatch')
            self.cache[name] = df
        return self.cache[name]

    def cutoff(self, mode):
        require(mode in MODES, f'Invalid CMC_DATA_MODE: {mode}')
        return pd.Timestamp(self.manifest[MODES[mode]])


def merge_exact(left, right, keys, name):
    unique(right, keys, name)
    result = left.merge(right, on=keys, how='left', validate='one_to_one', indicator=True)
    require(result['_merge'].eq('both').all(), f'{name}: missing referenced records')
    return result.drop(columns='_merge')


def course_features(snapshots, attempts):
    """Only grades available AND published at each snapshot's cutoff."""
    keys = snapshots[[*KEY, 'snapshot_cutoff']].rename(columns={'snapshot_term_index': 'term_index'})
    unique(keys, ['record_uid', 'term_index'], 'snapshot keys')
    a = attempts.merge(keys, on=['record_uid', 'term_index'], validate='many_to_one')
    a = a[(dates(a.available_at) <= dates(a.snapshot_cutoff)) &
          (dates(a.score_published_at) <= dates(a.snapshot_cutoff))]
    return a.groupby(['record_uid', 'term_index'], as_index=False).agg(
        ca_score10_min=('course_final_score10', 'min'), ca_exam10_min=('exam_final_score10', 'min'),
        ca_ineligible_exams=('exam_eligible', lambda s: int((~s).sum())),
        ca_unexcused_hours_sum=('unexcused_absent_hours', 'sum')).rename(columns={'term_index': 'snapshot_term_index'})


def asof_state(frame, states, when):
    unique(states, ['record_uid', 'effective_at'], 'state transitions')
    left = frame[['record_uid', when]].copy()
    left['_order'] = np.arange(len(left))
    left['_date'] = dates(left[when])
    right = states[['record_uid', 'effective_at', 'to_status']].copy()
    right['_effective'] = dates(right.effective_at)
    result = pd.merge_asof(left.sort_values('_date'), right.sort_values('_effective'),
                           left_on='_date', right_on='_effective', by='record_uid', direction='backward')
    require(result.to_status.notna().all(), 'Missing student state at cutoff')
    return result.sort_values('_order').to_status.to_numpy()


def certificates(frame, events):
    # Latest known status per student and requirement, not just presence of an event.
    unique(events, ['record_uid', 'requirement_type', 'available_at'], 'requirement events')
    results = [[] for _ in range(len(frame))]
    for kind, sub in events.groupby('requirement_type'):
        left = frame[['record_uid', 'snapshot_cutoff']].copy()
        left['_order'] = np.arange(len(left))
        left['_date'] = dates(left.snapshot_cutoff)
        right = sub[['record_uid', 'available_at', 'completion_status']].copy()
        right['_available'] = dates(right.available_at)
        m = pd.merge_asof(left.sort_values('_date'), right.sort_values('_available'),
                          left_on='_date', right_on='_available', by='record_uid', direction='backward')
        for idx in m.loc[m.completion_status.eq('VERIFIED_SYNTHETICALLY'), '_order']:
            results[idx].append(kind)
    return [', '.join(sorted(r)) or 'Chưa hoàn thành' for r in results]


def enrich(frame, source):
    frame = frame.copy()
    terms = source.table('term_academics')
    right = terms[['record_uid', 'term_index', 'available_at', *RUNTIME_TERM]].rename(columns={'term_index': 'snapshot_term_index', 'available_at': '_term_available'})
    frame = merge_exact(frame.drop(columns=RUNTIME_TERM, errors='ignore'), right, KEY, 'term academics')
    require((dates(frame._term_available) <= dates(frame.snapshot_cutoff)).all(), 'Future academic record')
    frame = frame.drop(columns='_term_available')
    frame['living_status'] = frame['living_status_at_cutoff']
    frame['verified_certificates'] = certificates(frame, source.table('academic_requirement_events'))
    frame['current_state'] = asof_state(frame, source.table('state_transitions'), 'snapshot_cutoff')
    return frame


def runtime_view(scored, source, mode='ASOF_2026_DEMO'):
    """Keep scores/ML features immutable; replace unsafe display fields only."""
    snapshots = source.table('prediction_snapshots')
    unique(scored, KEY, 'scored'); unique(snapshots, KEY, 'snapshots')
    require(len(scored) == len(snapshots), 'Scored/source population mismatch')
    common = snapshots.columns.intersection(scored.columns).difference(KEY)
    check = scored[[*KEY, *common]].merge(snapshots[[*KEY, *common]], on=KEY, validate='one_to_one', suffixes=('_old', '_src'))
    require(len(check) == len(scored), 'Snapshot keys differ from model artifact')
    for c in common:
        a, b = check[c+'_old'], check[c+'_src']
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            ok = np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)
        else:
            ok = a.fillna('').astype(str).eq(b.fillna('').astype(str)).all()
        require(ok, f'Scored/source field mismatch: {c}')
    cutoff = source.cutoff(mode)
    frame = scored.loc[dates(scored.snapshot_cutoff) <= cutoff].copy()
    frame = frame.drop(columns=['dropout_next_2_terms', 'exit_reason_for_evaluation', 'label_observed', 'first_dropout_event_id'], errors='ignore')
    # Runtime never includes labels, even when choosing FULL_SIMULATION.
    frame = enrich(frame, source)
    frame['data_mode'] = mode
    frame['data_as_of'] = cutoff.date().isoformat()
    latest = frame.sort_values('snapshot_term_index').drop_duplicates('record_uid', keep='last').copy()
    latest['state_as_of'] = asof_state(latest, source.table('state_transitions'), 'data_as_of')
    frame['state_as_of'] = frame.record_uid.map(latest.set_index('record_uid').state_as_of)
    return frame, latest


def visible_attempts(attempts, latest):
    keys = latest[['record_uid', 'snapshot_cutoff', 'snapshot_term_index']]
    a = attempts.merge(keys, on='record_uid', validate='many_to_one')
    return a.loc[(a.term_index <= a.snapshot_term_index) &
                 (dates(a.available_at) <= dates(a.snapshot_cutoff)) &
                 (dates(a.score_published_at) <= dates(a.snapshot_cutoff))].drop(columns=['snapshot_cutoff', 'snapshot_term_index'])
