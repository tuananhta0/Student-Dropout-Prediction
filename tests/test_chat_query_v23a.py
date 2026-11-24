
import unittest
from fastapi import HTTPException
import tests.test_chat_query_v2 as fixture


class QueryV23ATests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.QueryV2Tests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def test_focused_policy(self):
        cases = [
            ("Môn chứng chỉ có tính vào 24 tín chỉ nợ đọng không?", 18, "Không."),
            ("Học kỳ đầu tiên GPA dưới bao nhiêu thì bị cảnh báo?", 18, "0,8"),
            ("Từ học kỳ thứ hai trở đi, GPA dưới mức nào thì bị cảnh báo?", 18, "1,0"),
            ("GPA học kỳ được tính theo công thức nào?", 17, "GPA ="),
            ("GPA được làm tròn đến mấy chữ số thập phân?", 17, "hai chữ số"),
            ("Sinh viên cử nhân được học tối đa bao nhiêu học kỳ?", 3, "18 học kỳ"),
            ("GPA tích lũy 2,4 có đăng ký làm khóa luận được không?", 12, "Chưa đạt"),
            ("Em muốn học song bằng nhưng GPA 2,2 thì có được không?", 23, "40 tín chỉ"),
            ("Dùng tài liệu trong phòng thi bị xử lý thế nào?", 25, "hủy kết quả"),
            ("Hệ GM và hệ SM khác nhau thế nào về chuẩn ngoại ngữ?", 19, "B2"),
            ("Tự ý bỏ học thì bị tính bao nhiêu phần trăm học phí?", 12, "100%"),
        ]
        for question, article, expected in cases:
            with self.subTest(question=question):
                out = self.f.ask(question)
                self.assertEqual(out["intent"], "regulation")
                self.assertEqual(
                    {c["article"] for c in out["regulations"]},
                    {article}
                )
                visible = out["reply"].split("<details>")[0]
                self.assertIn(expected, visible)
                self.assertIn("Căn cứ:", visible)

    def test_cohort_scope(self):
        q = "Tỷ lệ sinh viên ngành Công nghệ thông tin bị cảnh báo học tập qua các khóa 2022–2025 thay đổi thế nào?"
        out = self.f.ask(q, "advisor")
        self.assertEqual(out["intent"], "analytics")
        self.assertIn("2025:", out["reply"])
        self.assertNotIn("2024:", out["reply"])

    def test_other_major_denied(self):
        q = "Tỷ lệ sinh viên ngành Quản trị kinh doanh bị cảnh báo học tập qua các khóa 2022–2025 thay đổi thế nào?"
        with self.assertRaises(HTTPException) as err:
            self.f.ask(q, "advisor")
        self.assertEqual(err.exception.status_code, 403)

    def test_school_scope_denied(self):
        q = "Tổng số sinh viên toàn trường là bao nhiêu?"
        with self.assertRaises(HTTPException) as err:
            self.f.ask(q, "advisor")
        self.assertEqual(err.exception.status_code, 403)
