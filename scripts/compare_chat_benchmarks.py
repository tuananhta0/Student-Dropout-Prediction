"""Compare paired offline chat runs by question ID; never certify semantics."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Empty or duplicate question IDs')
    return {r['id']:r for r in rows}


def compare(old,new):
    if set(old)!=set(new): raise ValueError('Question IDs differ')
    for k in old:
        if any(old[k][f]!=new[k][f] for f in ('question','role','type','expected_intent','source_reference','expected_articles')):
            raise ValueError(f'Question metadata differs: {k}')
    def metrics(rows):
        n=len(rows); statuses=Counter(r['http_status'] for r in rows)
        words=[int(r['visible_words']) for r in rows if r['http_status']=='200']
        labels=[r for r in rows if r['article_overlap'] in ('True','False')]
        return {'n':n,'http_status':dict(statuses),'actual_intents':dict(Counter(r['actual_intent'] or 'HTTP_'+r['http_status'] for r in rows)),
                'no_evidence_method_none':sum(r['method']=='none' for r in rows),
                'article_overlap_n':sum(r['article_overlap']=='True' for r in labels),'article_labeled_n':len(labels),
                'words_mean_http200':round(sum(words)/len(words),2) if words else None,'http200_over_200_words':sum(w>200 for w in words)}
    groups={}
    for name in sorted({r['type'] for r in old.values()}):
        ids=[k for k,r in old.items() if r['type']==name]
        groups[name]={'before':metrics([old[k] for k in ids]),'after':metrics([new[k] for k in ids]),
                      'intent_changes':sum(old[k]['actual_intent']!=new[k]['actual_intent'] for k in ids),
                      'status_changes':sum(old[k]['http_status']!=new[k]['http_status'] for k in ids)}
    return {'before':metrics(list(old.values())),'after':metrics(list(new.values())),'by_type':groups,
            'lost_article_overlap_ids':[k for k in old if old[k]['article_overlap']=='True' and new[k]['article_overlap']!='True'],
            'gained_article_overlap_ids':[k for k in old if old[k]['article_overlap']=='False' and new[k]['article_overlap']=='True'],
            'new_5xx_ids':[k for k in old if not old[k]['http_status'].startswith('5') and new[k]['http_status'].startswith('5')],
            'changed_http_ids':[k for k in old if old[k]['http_status']!=new[k]['http_status']],
            'semantic_review':'NOT_RUN','note':'Intent distribution and article overlap are diagnostic only. HTTP 403 can be correct. No semantic accuracy inferred.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    old,new=read(a.before),read(a.after); result=compare(old,new)
    result['inputs']={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('before_sha256',a.before),('after_sha256',a.after)]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return int(bool(result['new_5xx_ids']))

if __name__=='__main__':raise SystemExit(main())
