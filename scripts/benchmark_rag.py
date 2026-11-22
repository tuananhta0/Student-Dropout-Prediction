"""Benchmark bộ câu hỏi quy chế tiếng Việt; đo retrieval và kiểm tra trích dẫn, không tự chấm độ đúng LLM."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path
from core.retrieval import RegulationSearch

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


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cloud',action='store_true');ap.add_argument('--extended',action='store_true',help='Bộ câu diễn đạt khác');ap.add_argument('--cases',type=Path,help='JSON của người đánh giá độc lập: [[câu hỏi, Điều hoặc null], ...]');ap.add_argument('--probe',action='store_true',help='Kiểm tra một câu có Điều/Khoản, không gọi embedding hoặc ghi đè báo cáo');ap.add_argument('--output',type=Path,default=Path('data/kiem_tra_rag/bao_cao.json'));a=ap.parse_args()
    r=RegulationSearch()
    if a.probe:
        out=r.answer('Khoản 2 Điều 18',cloud=a.cloud)
        print(json.dumps({'phuong_phap':out['method'],'trang_thai_llm':out['generation_status'],
                          'co_can_cu':bool(out['regulations'])},ensure_ascii=False,indent=2))
        return 0 if out['method']=='gemini' else 1
    cases=json.loads(a.cases.read_text(encoding='utf-8-sig')) if a.cases else (EXTENDED if a.extended else CASES)
    if not isinstance(cases,(list,tuple)) or not cases or not all(isinstance(x,(list,tuple)) and len(x)==2 and isinstance(x[0],str) and x[0].strip() and (x[1] is None or type(x[1]) is int and 1<=x[1]<=27) for x in cases):
        ap.error('Bộ câu hỏi phải gồm các cặp [câu hỏi, số Điều hoặc null]')
    rows=[];rr=[];first=[];unknown=[]
    for q,expected in cases:
        out=r.answer(q,cloud=a.cloud);chunks=out['regulations'];articles=[c['article'] for c in chunks]
        rank=next((i+1 for i,v in enumerate(articles) if v==expected),0)
        if expected is None:unknown.append(not chunks)
        else:rr.append(1/rank if rank else 0);first.append(rank==1)
        cite_ok=all(f"#page={c['pages'].split('–')[0]}" in out['reply'] for c in chunks)
        rows.append({'cau_hoi':q,'dieu_mong_doi':expected,'dieu_truy_xuat':articles,'hang':rank,'co_trich_dan':cite_ok,'phuong_phap':out['method'],'dung_embedding':out.get('embedding_used',False),'trang_thai_llm':out['generation_status'],'loi_embedding':out.get('embedding_error'),'phan_hoi':out['reply']})
    report={'so_cau':len(rows),'so_cau_co_can_cu':len(rr),'hit_rate_at_5':sum(x>0 for x in rr)/len(rr) if rr else None,'mrr_at_5':sum(rr)/len(rr) if rr else None,'top1':sum(first)/len(first) if first else None,'tu_choi_ngoai_pham_vi':sum(unknown)/len(unknown) if unknown else None,'trich_dan_hop_le':all(x['co_trich_dan'] for x in rows),'so_cau_embedding':sum(x['dung_embedding'] for x in rows),'so_cau_gemini':sum(x['phuong_phap']=='gemini' for x in rows),'ghi_chu':'Bộ câu hỏi do người phát triển soạn, không phải tập kiểm thử độc lập. Phải đối chiếu ngữ nghĩa câu trả lời sinh với PDF; không có điểm tự chấm chất lượng LLM.','trang_thai_llm':dict(Counter(x['trang_thai_llm'] for x in rows)),'chi_tiet':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='chi_tiet'},ensure_ascii=False,indent=2))
    review=a.output.with_name(a.output.stem+'_doi_chieu.csv')
    with review.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['cau_hoi','dieu_mong_doi','dieu_da_truy_xuat','phuong_phap','phan_hoi_html','dung_ngu_nghia','day_du','ghi_chu_nguoi_cham'])
        for row in rows: writer.writerow([row['cau_hoi'],row['dieu_mong_doi'],row['dieu_truy_xuat'],row['phuong_phap'],row['phan_hoi'],'','',''])
    print('Báo cáo:',a.output,'; bảng đối chiếu thủ công:',review)

if __name__=='__main__':raise SystemExit(main())
