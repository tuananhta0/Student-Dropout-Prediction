"""Create auditable label corrections and stratified developer evaluation, never self-grade."""
import argparse
import ast
import csv
import json
import random
import re
from collections import defaultdict, Counter
from pathlib import Path
from scripts.report_provenance import provenance, sha256

# Only changes inspected on the corresponding original PDF page. No model-output label tuning.
CORRECTIONS={
    'SV-0034':('Quay lại học sau khi nghỉ có giữ nguyên mã sinh viên không?',9,'QĐ 22/2023, Điều 9, khoản 1b (PDF trang 11)'),
    'SV-0092':('Khóa luận đạt yêu cầu được lưu trữ bao lâu?',26,'QĐ 22/2023, Điều 26, khoản 2d (PDF trang 27)'),
    'SV-0093':('Khóa luận không đạt được lưu trữ bao lâu?',26,'QĐ 22/2023, Điều 26, khoản 2d (PDF trang 27)'),
    'SV-0504':('bảo vệ online được ko ạ',13,'QĐ 22/2023, Điều 13, khoản 2 (PDF trang 15)'),
}

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def articles(text):
    return sorted({int(v) for m in re.finditer(r'Điều\s+(\d+(?:\s*[,–-]\s*\d+)*)',text,re.I) for v in re.findall(r'\d+',m[1])})

def review(bank, results):
    orig={r['id']:r for r in bank}; got={r['id']:r for r in results}
    if len(orig)!=len(bank) or len(got)!=len(results) or set(orig)!=set(got):raise ValueError('Mismatched or duplicate question IDs')
    rows=[]
    for k,b in orig.items():
        r=got[k]
        if any(b[x]!=r[x] for x in ('question','role','type')) or b['source']!=r['source_reference']:raise ValueError('Metadata mismatch: '+k)
        src=b['source']; expected=articles(src); corr=CORRECTIONS.get(k)
        if corr:
            if b['question']!=corr[0]:raise ValueError('Question changed: '+k)
            src=corr[2];expected=[corr[1]]
        found=set(ast.literal_eval(r['displayed_articles'])); match=bool(found&set(expected)) if expected else ''
        rows.append({**r,'source_reference_reviewed':src,'expected_articles_reviewed':expected,'article_overlap_reviewed':match,'label_review_status':'CORRECTED_PDF' if corr else 'NOT_REVIEWED','reviewer_id':'','semantic_review_status':'NOT_RUN'})
    return rows


def choose(rows,n,seed=20261124):
    # Stratify by expected intent; sample selection does NOT depend on answer correctness.
    rng=random.Random(seed);groups=defaultdict(list)
    for r in rows:
        exp=r['expected_articles_reviewed'];q=r['question']
        if r['type']=='Quy chế' and len(exp)==1 and not re.search(r'\b(?:B[A-Z]{2}\d{6}|SIMV31_\d{6}|[A-Z]{2}\d{2}-\d{2,3})\b',q.upper()):groups[r['expected_intent']].append(r)
    for group in groups.values():rng.shuffle(group)
    chosen=[]
    while len(chosen)<n and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen)<n:chosen.append(groups[key].pop())
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bank',type=Path);p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--sample',type=int,default=48);a=p.parse_args()
    if not 1<=a.sample<=80:p.error('Sample must be 1..80 to preserve cloud budget')
    rows=review(read(a.bank),read(a.results));a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'label_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    selected=choose(rows,a.sample);cases=[[r['question'],r['expected_articles_reviewed'][0]] for r in selected]
    cases += [[q,None] for q in ['Thư viện mở cửa mấy giờ?','Lịch nghỉ lễ năm nay?','Giá một tín chỉ là bao nhiêu tiền?','Thời tiết hôm nay thế nào?','Khi nào đóng học phí kỳ tới?','Số tài khoản để chuyển học phí là gì?']]
    (a.output/'cloud_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'cloud_sample_manifest.json').write_text(json.dumps({'seed':20261124,'strategy':'round robin expected intent; random within stratum; 6 fixed refusal controls','sample_ids':[r['id'] for r in selected],'by_intent':dict(Counter(r['expected_intent'] for r in selected)),'independent':False,'cases_sha256':sha256(a.output/'cloud_cases.json')},ensure_ascii=False,indent=2),encoding='utf-8')
    # Blind-review instructions, not prefilled grades or fake independent questions.
    (a.output/'independent_README.txt').write_text('A team member not involved in coding RAG must prepare a new private JSON list [[question, article or null], ...]. Freeze it before seeing answers. Run scripts.benchmark_rag with --cases and --output (use --cloud for a separate paced run). Compare all answers to original PDF and grade all four 0/1 columns in the generated doi_chieu.csv. Run scripts.review_rag_semantics --reviewer MEMBER_CODE. For data questions, independently compare actual API results to scoped source tables. No automated PASS certifies semantics.\n',encoding='utf-8')
    report={'questions':len(rows),'corrected_pdf':sum(r['label_review_status']=='CORRECTED_PDF' for r in rows),'remaining_label_review':sum(r['label_review_status']=='NOT_REVIEWED' for r in rows),'original_labeled':sum(r['article_overlap'] in ('True','False') for r in rows),'original_overlap':sum(r['article_overlap']=='True' for r in rows),'reviewed_labeled':sum(r['article_overlap_reviewed']!='' for r in rows),'reviewed_overlap':sum(r['article_overlap_reviewed'] is True for r in rows),'cloud_sample_known':len(selected),'cloud_refusal_controls':6,'semantic_review':'NOT_RUN','provenance':provenance(bank_sha256=sha256(a.bank),results_sha256=sha256(a.results))}
    (a.output/'review_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
