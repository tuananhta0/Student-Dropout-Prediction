"""Tổng hợp chấm bằng tay cho MỌI câu (BM25, Gemini, fallback).

Không tự gán điểm. Người chấm đối chiếu văn bản PDF và điền 0/1, ghi lý do
khi không đạt; báo cáo không thay thế đánh giá mù độc lập.
"""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from scripts.report_provenance import provenance, sha256, read_report_origin

FIELDS = ('dung_ngu_nghia', 'day_du', 'trich_dan_dung', 'khong_bia_quy_dinh')


def summarize(path, reviewer):
    if not reviewer.strip():
        raise ValueError('Bắt buộc ghi mã người chấm; không tự xác nhận độc lập')
    with Path(path).open(encoding='utf-8-sig', newline='') as inp:
        rows = list(csv.DictReader(inp))
    if not rows:
        raise ValueError('Bảng chấm rỗng')
    scored=[]
    for i, row in enumerate(rows, 2):
        vals=[]
        for field in FIELDS:
            value=(row.get(field) or '').strip()
            if value not in ('0','1'):
                raise ValueError(f'Dòng {i}: phải chấm 0 hoặc 1 cho {field}')
            vals.append(int(value))
        if 0 in vals and not (row.get('ghi_chu_nguoi_cham') or '').strip():
            raise ValueError(f'Dòng {i}: trường hợp không đạt phải có lý do')
        scored.append((row.get('phuong_phap','unknown'), vals))
    counts=Counter(method for method,_ in scored)
    result={
        'nguoi_cham':reviewer.strip(),
        'so_cau':len(scored),
        'theo_phuong_phap':dict(counts),
        'ti_le_dat':{field:sum(vals[i] for _,vals in scored)/len(scored) for i,field in enumerate(FIELDS)},
        'dat_tat_ca_tieu_chi':sum(all(vals) for _,vals in scored)/len(scored),
        'gemini':{'so_cau':counts.get('gemini',0),'ti_le_dat':
                  (sum(all(v) for m,v in scored if m=='gemini')/counts['gemini'] if counts.get('gemini') else None)},
        'fallback':{'so_cau':len(scored)-counts.get('gemini',0)},
        'ghi_chu':'Điểm do người chấm nhập thủ công; chỉ được gọi độc lập khi người chấm không tham gia sửa thuật toán và bộ câu được giữ kín.',
    }
    result['provenance']=provenance(review_csv_sha256=sha256(path), benchmark_origin=read_report_origin(path))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    parser.add_argument('--reviewer',required=True)
    parser.add_argument('--output',type=Path,default=Path('data/kiem_tra_rag/ket_qua_cham_tat_ca.json'))
    args=parser.parse_args()
    data=summarize(args.csv,args.reviewer)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(data,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
