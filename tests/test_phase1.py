import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from chatbot_engine import CMCChatbotEngine
import train_model


class Phase1Tests(unittest.TestCase):

    def setUp(self):
        self.engine = CMCChatbotEngine.__new__(CMCChatbotEngine)
        self.engine.analytics = {}
        self.engine.df_attempts = None
        self.engine.course_names = {}
        self.engine.pipeline = None

    def test_no_backspace_control_bytes(self):
        source = Path("chatbot_engine.py").read_bytes()
        self.assertNotIn(bytes([8]), source)

    def test_student_id(self):
        self.assertEqual(
            self.engine.parse_flexible_query("Xem hồ sơ BIT250093")["student_id"],
            "BIT250093"
        )

    def test_record_uid(self):
        self.assertEqual(
            self.engine.parse_flexible_query("SIMV31_000007")["student_id"],
            "SIMV31_000007"
        )

    def test_class_code(self):
        self.assertEqual(
            self.engine.parse_flexible_query("Xem lớp IT25-09")["class_code"],
            "IT25-09"
        )

    def test_major_code(self):
        self.assertEqual(
            self.engine.parse_flexible_query("Ngành AI")["major_code"],
            "AI"
        )

    def test_term_filter(self):
        for query in ("Học kỳ 2", "HK2", "hoc ky 2"):
            with self.subTest(query=query):
                self.assertEqual(
                    self.engine.parse_flexible_query(query)["term_filter"], 2
                )

    def test_article(self):
        self.assertEqual(
            self.engine.parse_flexible_query("Điều 18")["articles"],
            ["18"]
        )

    def test_article_retrieval(self):
        result = self.engine.answer_query("Điều 18")
        self.assertIn("Điều 18", result["reply"])

    def test_keyword_retrieval(self):
        result = self.engine.answer_query("Rút học phần")
        self.assertIn("Điều 12", result["reply"])

    def test_no_model_fallback(self):
        with self.assertRaises(RuntimeError):
            self.engine.predict_what_if({"risk_probability": 0.2})

    def test_missing_course_data(self):
        self.assertEqual(self.engine.get_student_courses("X"), [])
        result = self.engine.render_course_grades_html("X", [])
        self.assertIn("Chưa có dữ liệu học phần", result)

    def test_no_matching_course_records(self):
        self.engine.df_attempts = pd.DataFrame([{
            "record_uid": "X",
            "term_index": 1,
            "course_id": "C001",
            "is_passed": True
        }])
        self.assertEqual(
            self.engine.get_student_courses("X", only_failed=True),
            []
        )

    def test_missing_course_scores_not_invented(self):
        self.engine.df_attempts = pd.DataFrame([{
            "record_uid": "X",
            "term_index": 1,
            "course_id": "C001",
            "registered_credits": 3,
            "process_score10": None,
            "exam_final_score10": None,
            "course_final_score10": None,
            "grade_letter": None,
            "is_passed": None,
            "exam_eligible": None,
        }])
        result = self.engine.get_student_courses("X")[0]
        self.assertIsNone(result["exam_score"])
        self.assertIsNone(result["grade_letter"])
        self.assertIsNone(result["is_passed"])
        self.assertEqual(result["course_name"], "C001")

    def test_required_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                CMCChatbotEngine(root)

    def test_missing_analytics(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in (
                "dropout_pipeline.joblib",
                "model_metadata.json",
                "student_snapshots_scored.csv.gz"
            ):
                (root / name).write_bytes(b"test")
            with self.assertRaisesRegex(
                FileNotFoundError, "analytics_summary"
            ):
                CMCChatbotEngine(root)

    def test_corrupt_model(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in (
                "dropout_pipeline.joblib",
                "model_metadata.json",
                "student_snapshots_scored.csv.gz",
                "analytics_summary.json"
            ):
                (root / name).write_bytes(b"test")

            with patch(
                "chatbot_engine.joblib.load",
                side_effect=ValueError("Corrupt artifact")
            ):
                with self.assertRaises(RuntimeError):
                    CMCChatbotEngine(root)

    def test_missing_training_input(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with patch.object(train_model, "DATASET_ROOT", root):
                with patch.object(train_model, "MAIN_RUN", root):
                    with self.assertRaises(FileNotFoundError):
                        train_model.find_file("missing_phase1_input.csv")

    def test_existing_training_input(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            path = root / "example.csv"
            path.write_text("id", encoding="utf-8")

            with patch.object(train_model, "DATASET_ROOT", root):
                with patch.object(train_model, "MAIN_RUN", root):
                    self.assertEqual(
                        train_model.find_file("example.csv"), path
                    )


if __name__ == "__main__":
    unittest.main()
