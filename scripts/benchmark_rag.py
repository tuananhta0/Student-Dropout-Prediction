"""Benchmark bộ câu hỏi quy chế tiếng Việt; đo retrieval và kiểm tra trích dẫn, không tự chấm độ đúng LLM."""
import argparse
import importlib
import hashlib
import os
import sys
from contextlib import contextmanager
from unittest.mock import patch
import csv
from collections import Counter
import json
from pathlib import Path
from core.retrieval import RegulationSearch
from scripts.report_provenance import provenance, sidecar

CASES=[
('Quy chế áp dụng cho ai?',1),('Mục tiêu đào tạo là gì?',2),('thời gian tối đa',3),('Môn bắt buộc không đạt thì sao?',4),('mỗi tiết bao nhiêu phút?',5),('trách nhiệm sinh viên',6),('cố vấn học tập có nhiệm vụ gì?',7),('kế hoạch đào tạo tháng 7',8),('lớp nhập học',9),('quy mô lớp học',10),('học trực tuyến',11),('rút học phần trong tuần đầu',12),('điểm liệt',13),('thi lại',14),('học cải thiện điểm',15),('phúc khảo',16),('công thức GPA',17),('cảnh báo học tập',18),('tiếng Anh B2',19),('điều kiện tốt nghiệp',20),('bảo lưu',21),('chuyển ngành',22),('học song ngành',23),('trao đổi sinh viên',24),('cấm thi',25),('lưu trữ khóa luận',26),('hiệu lực thi hành',27),('Khoản 2 Điều 18',18),('Học phí rút môn',12),('hạ hạng tốt nghiệp',20),('thiên văn',None),('Trái đất có bao nhiêu mặt trăng',None),('Điều 99',None),('xyzxyzxyz',None)]

# Tập hồi quy diễn đạt khác benchmark cũ. Do người phát triển soạn, không phải bộ độc lập mù.
EXTENDED=[
('Làm sao xin chấm lại bài thi?',16),('Học hết bao nhiêu năm thì phải ra trường?',3),
('Bao nhiêu lần bị nhắc nhở liên tiếp thì bị đuổi học?',18),('Chuyển sang trường đại học khác',22),
('Em có thể bảo lưu khi bị ốm không?',21),('Điều kiện học phần tiên quyết của em là gì?',3),
('GPA bao nhiêu để được làm đồ án?',12),('Đăng ký tín chỉ tối đa bao nhiêu?',12),
('Nghỉ không phép quá 20 phần trăm thì sao?',25),('Không đạt học phần bắt buộc phải làm gì?',4),
('Thời gian nghỉ vì bệnh có tính vào tiến độ không?',21),('Luận văn phải được lưu bao lâu?',26),
('Có được đăng ký ngành thứ hai không?',23),('Thi cuối kỳ 2 điểm thì có qua môn không?',13),
('Em có thể đóng tiền bằng thẻ ngân hàng không?',None),('Cách đăng ký vay tiền ngân hàng?',None),
('Mức phí gửi xe ở trường là bao nhiêu?',None),('Học phí bao nhiêu một tháng?',None)]


@contextmanager
def endpoint(engine=None):
    """CI kiểm thử API thật với hồ sơ fixture, không phụ thuộc release hoặc Gemini."""
    from fastapi.testclient import TestClient
    from tests.test_security import FakeEngine
    original=sys.modules.pop('app',None)
    try:
        with patch.dict(os.environ,{'CMC_RAG_CLOUD':'0'}), patch('core.runtime_config.configure_runtime',return_value='fixture'),patch('core.chatbot_engine.CMCChatbotEngine',return_value=FakeEngine() if engine is None else engine):
            module=importlib.import_module('app')
            with TestClient(module.app) as client:
                def invoke(q):
                    # Xóa context giữa các câu đơn lẻ; không đo rate limit như chất lượng RAG.
                    module.contexts[token.json()['access_token']]={}
                    res=client.post('/api/chat',headers=headers,json={'message':q})
                    res.raise_for_status()
                    out=res.json()
                    return out
                token=client.post('/api/login',json={'username':'BIT250093','password':'BIT250093'})
                token.raise_for_status();headers={'Authorization':'Bearer '+token.json()['access_token']}
                with patch.object(module.chat_limit,'check'), patch.object(module,'contexts',{token.json()['access_token']: {}}):
                    yield invoke
    finally:
        sys.modules.pop('app',None)
        if original is not None: sys.modules['app']=original

REGRESSION=[
 ('Thời gian tối đa',3),('Học cải thiện điểm',15),('Đăng ký tín chỉ tối đa bao nhiêu?',12),
 ('Học phí mỗi tín chỉ là bao nhiêu?',None),('Trường có câu lạc bộ bóng đá không?',None),
 ('Vắng 25% có được thi không?',25),('Nghỉ học do bệnh có cần giấy xác nhận không?',21),
 ('Tôi muốn nghỉ một năm để chăm người thân',21),('Em có nợ 25 tín chỉ theo Điều 16 thì sao?',18),
 ('Tự xin thôi học',21),('Quyền lợi sau thôi học',20),
 ('Lịch nghỉ lễ của sinh viên?',None),('Thời hạn nộp học phí kỳ tới?',None),
 ('Thư viện trường mở cửa mấy giờ?',None),('Em có 3 lần cảnh báo liên tiếp có bị thôi học không?',18),
 ('Điều kiện học lớp cải thiện khi đạt C?',15),('Học phí cao quá em muốn nghỉ học một năm được không?',21),
 ('Em nợ 25 tín chỉ chứng chỉ thì sao?',18),('Em thi cuối kỳ 4 điểm có qua môn không?',13)
]

def quality_pass(report, top1=.95):
    """Tập thiếu nhóm câu không bị coi là có điểm 0; vẫn phải có ít nhất một câu."""
    return bool(report['so_cau'] and report['trich_dan_hop_le'] and
        (not report.get('so_cau_co_can_cu') or report.get('ty_le_dieu_mong_doi_duoc_hien_thi') is not None) and
        all(k in report and (report[k] is None or report[k]>=v) for k,v in [('hit_rate_at_5',.95),('mrr_at_5',.95),('top1',top1),('tu_choi_ngoai_pham_vi',1.),('ty_le_dieu_mong_doi_duoc_hien_thi',1.)]))

def cloud_quality(rows):
    """Đo generation riêng: không coi fallback hoặc ID hợp lệ là LLM thành công."""
    known=[r for r in rows if r['dieu_mong_doi'] is not None]
    ok=sum(r['phuong_phap']=='gemini' and r['trang_thai_llm']=='OK' for r in known)
    rate=ok/len(known) if known else None
    return {'so_cau_gemini_mong_doi':len(known),'ty_le_gemini_thanh_cong':rate,
            'cloud_quality_gate':bool(rate is not None and rate>=.90)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cloud',action='store_true');ap.add_argument('--e2e',action='store_true',help='Kiểm thử API nội bộ với fixture, cloud tắt');ap.add_argument('--all',action='store_true',help='Gộp các bộ hồi quy');ap.add_argument('--extended',action='store_true',help='Bộ câu diễn đạt khác');ap.add_argument('--cases',type=Path,help='JSON của người đánh giá độc lập: [[câu hỏi, Điều hoặc null], ...]');ap.add_argument('--probe',action='store_true',help='Kiểm tra một câu có Điều/Khoản, không gọi embedding hoặc ghi đè báo cáo');ap.add_argument('--output',type=Path,default=Path('data/kiem_tra_rag/bao_cao.json'));a=ap.parse_args()
    if a.e2e and (a.cloud or a.probe): ap.error('--e2e chỉ hỗ trợ offline, không dùng --probe')
    r=RegulationSearch()
    if a.probe:
        out=r.answer('Khoản 2 Điều 18',cloud=a.cloud)
        print(json.dumps({'phuong_phap':out['method'],'trang_thai_llm':out.get('generation_status','NOT_REQUESTED'),
                          'co_can_cu':bool(out['regulations'])},ensure_ascii=False,indent=2))
        return 0 if out['method']=='gemini' else 1
    cases=json.loads(a.cases.read_text(encoding='utf-8-sig')) if a.cases else (list(dict.fromkeys(CASES+EXTENDED+(REGRESSION if a.e2e else []))) if a.all else EXTENDED if a.extended else CASES)
    if not isinstance(cases,(list,tuple)) or not cases or not all(isinstance(x,(list,tuple)) and len(x)==2 and isinstance(x[0],str) and x[0].strip() and (x[1] is None or type(x[1]) is int and 1<=x[1]<=27) for x in cases):
        ap.error('Bộ câu hỏi phải gồm các cặp [câu hỏi, số Điều hoặc null]')
    rows=[];rr=[];first=[];unknown=[]
    from contextlib import nullcontext
    with (endpoint() if a.e2e else nullcontext(lambda q:r.answer(q,cloud=a.cloud))) as invoke:
      for q,expected in cases:
        out=invoke(q);chunks=out['regulations'];articles=out.get('retrieved_articles',[c['article'] for c in out.get('retrieved',chunks)])
        rank=next((i+1 for i,v in enumerate(articles[:5]) if v==expected),0)
        if expected is None:unknown.append(not chunks and out.get('intent','regulation') in ('regulation','none','clarify'))
        else:rr.append(1/rank if rank else 0);first.append(rank==1)
        cite_ok=all(f"/regulations/{c['document']}#page={c['pages'].split('–')[0]}" in out['reply'] for c in chunks)
        rows.append({'cau_hoi':q,'dieu_mong_doi':expected,'dieu_truy_xuat':articles,'dieu_hien_thi':[c['article'] for c in chunks],'hang':rank,'co_trich_dan':cite_ok,'phuong_phap':out['method'],'dung_embedding':out.get('embedding_used',False),'trang_thai_llm':out.get('generation_status','NOT_REQUESTED'),'loi_embedding':out.get('embedding_error'),'intent':out.get('intent','regulation'),'phan_hoi':out['reply']})
    report={'so_cau':len(rows),'so_cau_co_can_cu':len(rr),'hit_rate_at_5':sum(x>0 for x in rr)/len(rr) if rr else None,'mrr_at_5':sum(rr)/len(rr) if rr else None,'top1':sum(first)/len(first) if first else None,'tu_choi_ngoai_pham_vi':sum(unknown)/len(unknown) if unknown else None,'trich_dan_hop_le':all(x['co_trich_dan'] for x in rows),'so_cau_embedding':sum(x['dung_embedding'] for x in rows),'so_cau_gemini':sum(x['phuong_phap']=='gemini' for x in rows),'ghi_chu':'Bộ câu hỏi do người phát triển soạn, không phải tập kiểm thử độc lập. Phải đối chiếu ngữ nghĩa câu trả lời sinh với PDF; không có điểm tự chấm chất lượng LLM.','trang_thai_llm':dict(Counter(x['trang_thai_llm'] for x in rows)),'chi_tiet':rows}
    report['ty_le_dieu_mong_doi_duoc_hien_thi']=sum(row['dieu_mong_doi'] in row['dieu_hien_thi'] for row in rows if row['dieu_mong_doi'] is not None)/len(rr) if rr else None
    report['quality_gate']=quality_pass(report)
    report['e2e_fixture']=a.e2e
    report['nguon_bo_cau']='do người đánh giá cung cấp (cần xác nhận độc lập)' if a.cases else 'hồi quy do người phát triển soạn'
    report['cloud_requested']=a.cloud
    report['provenance']=provenance(corpus_sha256=r.corpus_sha, cases_sha256=hashlib.sha256(json.dumps(cases,ensure_ascii=False).encode()).hexdigest(), cloud_requested=a.cloud, llm_model=os.getenv('CMC_LLM_MODEL','gemini-3.1-flash-lite') if a.cloud else None, embedding_model=os.getenv('CMC_EMBED_MODEL','gemini-embedding-001') if a.cloud else None)
    if a.cloud:
        report.update(cloud_quality(rows))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='chi_tiet'},ensure_ascii=False,indent=2))
    review=a.output.with_name(a.output.stem+'_doi_chieu.csv')
    with review.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['cau_hoi','dieu_mong_doi','dieu_da_truy_xuat','phuong_phap','phan_hoi_html','dung_ngu_nghia','day_du','ghi_chu_nguoi_cham'])
        for row in rows: writer.writerow([row['cau_hoi'],row['dieu_mong_doi'],row['dieu_truy_xuat'],row['phuong_phap'],row['phan_hoi'],'','',''])
    sidecar(review,report['provenance'])
    print('Báo cáo:',a.output,'; bảng đối chiếu thủ công:',review)
    return 0 if report['quality_gate'] and (not a.cloud or report['cloud_quality_gate']) else 1

if __name__=='__main__':raise SystemExit(main())
