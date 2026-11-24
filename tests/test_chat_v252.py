"""Developer regressions for precise public clauses and ambiguous requests."""
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from core.intent_router import route
from core.retrieval import public_focus
from core.security import Principal
import tests.test_chat_query_v2 as fixture


class ChatV252(unittest.TestCase):
    def setUp(self):
        self.f=fixture.QueryV2Tests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.r=self.f.r

    def test_precise_retrieval(self):
        cases=[
            ('Chưa học hết kỳ đầu thì có được bảo lưu vì lý do cá nhân không?',21),
            ('GPA học kỳ khác GPA tích lũy như thế nào?',17),
            ('Tích lũy 82 tín chỉ thuộc trình độ năm thứ mấy?',17),
            ('Hết thời gian đào tạo mà chưa có chứng chỉ quốc phòng thì sao?',20),
            ('Chưa đạt chuẩn ngoại ngữ có tốt nghiệp được không?',20),
            ('Vắng thi không có lý do thì nhận điểm gì?',14),
            ('Giảng viên gửi danh sách không đủ điều kiện dự thi trong mấy ngày?',14),
            ('Học song bằng cần học lực thế nào?',23),
            ('Chương trình kỹ sư có bao nhiêu tín chỉ?',3),
            ('Có được học trực tuyến 100% chương trình không?',11),
            ('Không đi học môn này nữa thì điểm sao?',12),
        ]
        for q,a in cases:
            with self.subTest(q=q):
                self.assertEqual(self.r.retrieve(q)[0]['article'],a)
                out=self.f.ask(q);self.assertEqual(out['intent'],'regulation');self.assertIn(a,{c['article'] for c in out['regulations']})

    def test_ambiguous_penalty(self):
        for q in ('lần 2 bị trừ bao nhiêu điểm','lần thứ hai bị trừ bao nhiêu điểm?'):
            out=self.f.ask(q);self.assertEqual(out['intent'],'clarify');self.assertFalse(out['regulations']);self.assertIn('hành vi nào',out['reply'])
        self.assertFalse(self.r.retrieve('lần 2 bị trừ bao nhiêu điểm'))

    def test_private_data_not_projected(self):
        q='GPA học kỳ khác GPA tích lũy thế nào? bí mật xyzsecret'
        focus=public_focus(q);self.assertNotIn('xyzsecret',focus)
        self.assertIn('phân biệt phạm vi',focus)
        with self.assertRaises(HTTPException):
            self.f.ask('GPA học kỳ khác GPA tích lũy thế nào? Xem hồ sơ BBA250001')

    def test_no_embedding_for_precise_clause(self):
        with patch.object(self.r,'_semantic',side_effect=AssertionError('unnecessary embedding')):
            self.assertEqual(self.r.retrieve('Có được học online toàn bộ chương trình không?',cloud=True)[0]['article'],11)

    def test_unsupported_still_refused(self):
        self.assertFalse(self.r.retrieve('Giá một tín chỉ là bao nhiêu tiền?'))
        self.assertFalse(self.r.retrieve('Lịch nghỉ lễ năm nay?'))
        self.assertFalse(self.r.retrieve('Học online toàn bộ chương trình giá bao nhiêu tiền? Số tài khoản chuyển khoản?'))
