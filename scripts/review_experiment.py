"""Đánh giá bổ sung trên xác suất đã lưu; không fit, không thay model/ngưỡng."""
import argparse
import json
from pathlib import Path
import pandas as pd
from core.data_pipeline import Dataset, MAIN, require
from scripts.release_model import verify_pair
from scripts.train_model import TARGET, evaluate, calibration, fairness, split_at_freeze


def bootstrap_students(test, threshold, n_resamples=200, seed=2026):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    if n_resamples==0: return None
    if not 20<=n_resamples<=2000: raise ValueError('Số lượt bootstrap phải từ 20 đến 2000 hoặc 0')
    codes,students=pd.factorize(test.record_uid)
    y=test[TARGET].to_numpy(dtype=int);p=test.risk_probability.to_numpy(dtype=float)
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(n_resamples):
        weights=np.bincount(rng.integers(0,len(students),len(students)),minlength=len(students))[codes]
        positives=weights[y==1].sum();negatives=weights[y==0].sum()
        if not positives or not negatives: continue
        samples.append([average_precision_score(y,p,sample_weight=weights),roc_auc_score(y,p,sample_weight=weights),weights[(y==1)&(p>=threshold)].sum()/positives])
    return {'seed':seed,'n_sinh_vien':len(students),'n_luot_hop_le':len(samples),'ci95':dict(zip(('AP','ROC_AUC','Recall'),np.percentile(samples,[2.5,97.5],axis=0).T.tolist())) if samples else None,'ghi_chu':'Bootstrap cụm theo sinh viên trên tập test đã lưu; không bao hàm sai số do huấn luyện hoặc sai lệch dữ liệu mô phỏng.'}

def review(scored, students, meta, original, n_resamples=0):
    _,_,test=split_at_freeze(scored,meta['model_available_as_of'])
    test=test[test.snapshot_term_index.isin(meta['supported_terms'])].copy()
    threshold=meta['threshold']
    def metrics(g): return evaluate(g[TARGET],g.risk_probability,threshold)
    def grouped(frame,col):
        return {str(k):metrics(g) if len(g)>=30 and g[TARGET].nunique()==2 else {'n':len(g),'positive':int(g[TARGET].sum()),'note':'Mẫu/nhãn quá ít để diễn giải'} for k,g in frame.groupby(col,dropna=False)}
    bins=calibration(test[TARGET],test.risk_probability)
    ece=sum(b['n']*abs(b['xac_suat_trung_binh']-b['ty_le_thuc_te']) for b in bins)/len(test)
    first=test.sort_values(['record_uid','snapshot_term_index']).drop_duplicates('record_uid')
    later=scored[scored.label_observed & scored[TARGET].notna() & scored.snapshot_cutoff.ge(meta['model_available_as_of']) & scored.snapshot_term_index.isin(meta['supported_terms'])]
    actual=metrics(test)
    require(len(test)==original['so_luong_tap']['test'],'Tập test không khớp báo cáo gốc')
    require(abs(actual['average_precision']-meta['test_pr_auc'])<1e-7,'AP khác metadata')
    return {'test_snapshot':actual,'test_sinh_vien_snapshot_dau_tien':metrics(first),
        'test_theo_hoc_ky':grouped(test,'snapshot_term_index'),
        'sau_moc_dong_bang_theo_khoa':grouped(later,'cohort_year'),
        'fairness_test':fairness(test,test.risk_probability,threshold,students),
        'fairness_stress_da_luu':original.get('stress_fairness'),
        'calibration':bins,'ece':ece,'bootstrap_test':bootstrap_students(test,threshold,n_resamples),
        'ghi_chu':['Một sinh viên: snapshot quan sát được đầu tiên trong tập test; không đo khả năng phát hiện ít nhất một lần trong cả quãng học.',
                    'Khóa 2022–2024 đã tham gia train/validation: đánh giá sau cutoff chỉ là theo dõi thời gian, không phải test độc lập theo sinh viên.',
                    'ECE phụ thuộc cách chia bin; không chứng minh calibration tốt ngoài dữ liệu mô phỏng.',
                    'Chênh lệch nhóm không chứng minh phân biệt đối xử hoặc quan hệ nhân quả; đối chiếu kích thước mẫu, tỷ lệ nền, recall, precision và Brier.',
                    'What-If học phí là độ nhạy của model với giả định đầu vào, không phải hiệu quả can thiệp thật.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bootstrap',type=int,default=200,help='Số lượt bootstrap theo sinh viên (0 để tắt)');p.add_argument('--output',type=Path,default=Path('data/kiem_tra_ml/bao_cao.json'));a=p.parse_args()
    from core.runtime_config import configure_runtime
    model=Path(configure_runtime());source=Dataset();meta=verify_pair(source.run.parents[2],model)
    scored=pd.read_csv(model/'student_snapshots_scored.csv.gz',low_memory=False)
    original=json.loads((model/'bao_cao_thuc_nghiem.json').read_text(encoding='utf-8'))
    report=review(scored,source.table('students'),meta,original,a.bootstrap)
    from scripts.report_provenance import provenance, sha256
    report["provenance"]=provenance(model_version=meta["version"],model_metadata_sha256=sha256(model/"model_metadata.json"),scored_sha256=sha256(model/"student_snapshots_scored.csv.gz"),dataset_manifest_sha256=meta.get("dataset_manifest_sha256"),bootstrap=a.bootstrap)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'test_snapshot':report['test_snapshot'],'test_sinh_vien_snapshot_dau_tien':report['test_sinh_vien_snapshot_dau_tien'],'ece':report['ece']},ensure_ascii=False,indent=2)); print('Báo cáo:',a.output)

if __name__=='__main__':main()
