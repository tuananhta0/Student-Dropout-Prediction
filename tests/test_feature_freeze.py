import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

from scripts import benchmark_rag as b
from scripts import review_policy


class FakeClock:
    def __init__(self, now):
        self.now = now
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


class FreezeTests(unittest.TestCase):
    def pace(self, calls, now):
        clock = FakeClock(now)
        queue = deque(calls)
        before = list(queue)
        with patch.object(b.cloud_runtime, '_cloud_calls', queue):
            result = b.wait_for_cloud_budget(clock.time, clock.sleep)
        self.assertEqual(list(queue), before)
        return result, clock

    def test_spare_budget_no_wait(self):
        result, clock = self.pace(range(24), 30)
        self.assertEqual(result, (0.0, 0))
        self.assertEqual(clock.sleeps, [])

    def test_full_minute_waits_for_six_slots(self):
        result, clock = self.pace(range(30), 30)
        self.assertAlmostEqual(result[0], 35.5)
        self.assertEqual(result[1], 1)
        self.assertEqual(len(clock.sleeps), 1)

    def test_expired_calls_do_not_consume_minute(self):
        result, _ = self.pace([0] * 30, 60)
        self.assertEqual(result, (0.0, 0))

    def test_daily_exhaustion_does_not_wait_for_hours(self):
        result, clock = self.pace(
            [0] * b.cloud_runtime.CLOUD_LIMIT_DAY, 1
        )
        self.assertEqual(result, (0.0, 0))
        self.assertEqual(clock.sleeps, [])

    def test_offline_does_not_invoke_pacer(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp) / 'cases.json'
            output = Path(tmp) / 'report.json'
            cases.write_text(json.dumps([['xyzxyzxyz', None]]))
            args = ['benchmark_rag', '--cases', str(cases),
                    '--output', str(output)]

            with patch('sys.argv', args), patch.object(
                b, 'wait_for_cloud_budget',
                side_effect=AssertionError('Offline pacing')
            ):
                self.assertEqual(b.main(), 0)

            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertNotIn('cloud_pacing', report)

    def test_cloud_records_pacing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp) / 'cases.json'
            output = Path(tmp) / 'report.json'
            cases.write_text(json.dumps([['xyzxyzxyz', None]]))
            args = ['benchmark_rag', '--cloud', '--cases', str(cases),
                    '--output', str(output)]

            with patch('sys.argv', args), patch.object(
                b, 'wait_for_cloud_budget', return_value=(2.5, 1)
            ) as pacing:
                # No known-answer questions: cloud gate must not pass.
                self.assertEqual(b.main(), 1)
                pacing.assert_called_once()

            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['cloud_pacing']['wait_seconds'], 2.5)
            self.assertFalse(report['cloud_quality_gate'])

    def test_stress_policy_requires_explicit_output(self):
        with self.assertRaises(SystemExit) as err:
            review_policy.main(['--stress-run', 'example'])
        self.assertEqual(err.exception.code, 2)

    def test_stress_cannot_overwrite_main_policy_output(self):
        with self.assertRaises(SystemExit) as err:
            review_policy.main([
                '--stress-run', 'example',
                '--output', str(review_policy.ROOT / 'data/doi_chieu_quy_che')
            ])
        self.assertEqual(err.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
