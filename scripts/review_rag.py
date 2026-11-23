"""Tổng hợp chấm thủ công độc lập; không tự chấm nội dung bằng LLM."""
import argparse
import csv
import json
from pathlib import Path

def summarize(path, reviewer):
    if not reviewer.strip(): raise ValueError('Cần mã người đánh giá')
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    if not rows: raise ValueError('Bảng đối chiếu rỗng')
    generated=[row for row in rows if row.get('phuong_phap')=='gemini']
    def score(row,key):
        v=row.get(key,'').strip()
        if v not in ('0','1'): raise ValueError(f"Cần chấm 0 hoặc 1 cho {key}: {row.get('cau_hoi','')[:80]}")
        return int(v)
    ratings=[(score(row,'dung_ngu_nghia'),score(row,'day_du')) for row in generated]
    if not ratings: raise ValueError('Không có câu Gemini để đánh giá')
    return {'nguoi_cham':reviewer,'so_cau':len(rows),'so_cau_gemini':len(ratings),
            'dung_ngu_nghia':sum(a for a,b in ratings)/len(ratings),'day_du':sum(b for a,b in ratings)/len(ratings),
            'dat_ca_hai':sum(a and b for a,b in ratings)/len(ratings),
            'ghi_chu':'Kết quả chấm thủ công; chỉ đánh giá câu Gemini, không tính fallback như câu sinh thành công. Người chấm phải độc lập với người chọn/điều chỉnh thuật toán.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv',type=Path);p.add_argument('--reviewer',required=True)
    p.add_argument('--output',type=Path,default=Path('data/kiem_tra_rag/danh_gia_doc_lap.json'))
    a=p.parse_args(); result=summarize(a.csv,a.reviewer)
    from scripts.report_provenance import provenance, sha256, read_report_origin
    result['provenance']=provenance(review_csv_sha256=sha256(a.csv),benchmark_csv_origin=read_report_origin(a.csv))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
