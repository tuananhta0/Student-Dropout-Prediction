"""Benchmark bộ câu hỏi quy chế tiếng Việt; đo retrieval và kiểm tra trích dẫn, không tự chấm độ đúng LLM."""
import argparse
from collections import Counter
import json
from pathlib import Path
from core.retrieval import RegulationSearch

CASES=[
('Quy chế áp dụng cho ai?',1),('Mục tiêu đào tạo là gì?',2),('thời gian tối đa',3),('Môn bắt buộc không đạt thì sao?',4),('mỗi tiết bao nhiêu phút?',5),('trách nhiệm sinh viên',6),('cố vấn học tập có nhiệm vụ gì?',7),('kế hoạch đào tạo tháng 7',8),('lớp nhập học',9),('quy mô lớp học',10),('học trực tuyến',11),('rút học phần trong tuần đầu',12),('điểm liệt',13),('thi lại',14),('học cải thiện điểm',15),('phúc khảo',16),('công thức GPA',17),('cảnh báo học tập',18),('tiếng Anh B2',19),('điều kiện tốt nghiệp',20),('bảo lưu',21),('chuyển ngành',22),('học song ngành',23),('trao đổi sinh viên',24),('cấm thi',25),('lưu trữ khóa luận',26),('hiệu lực thi hành',27),('Khoản 2 Điều 18',18),('Học phí rút môn',12),('hạ hạng tốt nghiệp',20),('thiên văn',None),('Trái đất có bao nhiêu mặt trăng',None),('Điều 99',None),('xyzxyzxyz',None)]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cloud',action='store_true');ap.add_argument('--probe',action='store_true',help='Kiểm tra một câu có Điều/Khoản, không gọi embedding hoặc ghi đè báo cáo');ap.add_argument('--output',type=Path,default=Path('data/kiem_tra_rag/bao_cao.json'));a=ap.parse_args()
    r=RegulationSearch()
    if a.probe:
        out=r.answer('Khoản 2 Điều 18',cloud=a.cloud)
        print(json.dumps({'phuong_phap':out['method'],'trang_thai_llm':out['generation_status'],
                          'co_can_cu':bool(out['regulations'])},ensure_ascii=False,indent=2))
        return 0 if out['method']=='gemini' else 1
    rows=[];rr=[];first=[];unknown=[]
    for q,expected in CASES:
        out=r.answer(q,cloud=a.cloud);chunks=out['regulations'];articles=[c['article'] for c in chunks]
        rank=next((i+1 for i,v in enumerate(articles) if v==expected),0)
        if expected is None:unknown.append(not chunks)
        else:rr.append(1/rank if rank else 0);first.append(rank==1)
        cite_ok=all(f"#page={c['pages'].split('–')[0]}" in out['reply'] for c in chunks)
        rows.append({'cau_hoi':q,'dieu_mong_doi':expected,'dieu_truy_xuat':articles,'hang':rank,'co_trich_dan':cite_ok,'phuong_phap':out['method'],'dung_embedding':out.get('embedding_used',False),'trang_thai_llm':out['generation_status'],'loi_embedding':out.get('embedding_error'),'phan_hoi':out['reply']})
    report={'so_cau':len(rows),'so_cau_co_can_cu':len(rr),'hit_rate_at_5':sum(x>0 for x in rr)/len(rr),'mrr_at_5':sum(rr)/len(rr),'top1':sum(first)/len(first),'tu_choi_ngoai_pham_vi':sum(unknown)/len(unknown),'trich_dan_hop_le':all(x['co_trich_dan'] for x in rows),'so_cau_embedding':sum(x['dung_embedding'] for x in rows),'so_cau_gemini':sum(x['phuong_phap']=='gemini' for x in rows),'ghi_chu':'Bộ câu hỏi do người phát triển soạn, không phải tập kiểm thử độc lập. Phải đối chiếu ngữ nghĩa câu trả lời sinh với PDF; không có điểm tự chấm chất lượng LLM.','trang_thai_llm':dict(Counter(x['trang_thai_llm'] for x in rows)),'chi_tiet':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='chi_tiet'},ensure_ascii=False,indent=2))
    print('Báo cáo:',a.output)

if __name__=='__main__':raise SystemExit(main())
