"""Xuất chỉ mục từ các đoạn đã đối chiếu ảnh PDF gốc; không OCR giả hoặc sửa artifact ML."""
import json
from pathlib import Path
from core.retrieval import RegulationSearch, PDF, PDF_SHA

if __name__=='__main__':
    r=RegulationSearch()
    out=Path('data/kiem_tra_rag/regulations_index.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'document':PDF,'pdf_sha256':PDF_SHA,'corpus_sha256':r.corpus_sha,'kind':'Bản tóm lược đối chiếu ảnh trang','chunks':r.chunks},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Đã xuất {len(r.chunks)} đoạn của 27 điều: {out}')
