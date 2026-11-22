"""Thực nghiệm và xuất mô hình nghiên cứu mới vào thư mục riêng, không ghi đè baseline."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from core.data_pipeline import (Dataset, ROOT, KEY, EXTRA_FEATURES, course_features, merge_exact,
                                runtime_view, academic_view, visible_attempts, digest, TABLES, require)
from core.prediction import model_inputs, risk_level
from core.analytics import build_analytics_summary
from core.data_pipeline import main_run
from scripts.audit_dataset import audit

MAIN_RUN = main_run()
DATASET_ROOT = MAIN_RUN.parents[2]
OUTPUT_DIR = ROOT / "artifacts_model"

def find_file(rel_path: str, fallback_name: str = "") -> Path:
    candidates = [
        MAIN_RUN / rel_path,

    ]

    for candidate in candidates:
        if candidate.is_file():
            if candidate.stat().st_size == 0:
                raise ValueError(f"File đầu vào rỗng: {candidate}")
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Không tìm thấy file dữ liệu: {rel_path}. Đã kiểm tra: {checked}"
    )



SEED = 221026
FREEZE = '2027-02-03'
TARGET = 'dropout_next_2_terms'
ADMIN = 'administrative_action_eligible_at_cutoff'


def join_data(source):
    x = source.table('prediction_snapshots')
    y = source.table('snapshot_labels_FULL_SIMULATION')
    require(y.label_observed.eq(y[TARGET].notna()).all(), 'Nhãn thiếu không nhất quán')
    df = merge_exact(x,y[KEY+[TARGET,'label_observed','label_observation_end','exit_reason_for_evaluation']],KEY,'nhãn')
    df = merge_exact(df,course_features(x,source.table('course_attempts')),KEY,'đặc trưng học phần')
    return df


def feature_meta(source, df):
    registry = source.table('feature_registry')
    cols = registry.loc[registry.role.eq('FEATURE_ALLOWED_ASOF'),'feature_name'].tolist()+EXTRA_FEATURES
    require(not set(cols) & {'student_id','record_uid','cohort_year',TARGET,'snapshot_cutoff','feature_source_latest_at','label_observed','exit_reason_for_evaluation'}, 'Đặc trưng bị cấm')
    require(len(set(cols))==len(cols) and set(cols)<=set(df), 'Registry không hợp lệ')
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    return {'feature_cols':cols,'cat_cols':cat,'num_cols':[c for c in cols if c not in cat], 'categorical_missing_token':'__MISSING__'}


def subset_meta(meta, cols):
    return dict(meta,feature_cols=list(cols),cat_cols=[c for c in meta['cat_cols'] if c in cols],num_cols=[c for c in meta['num_cols'] if c in cols])


def prep(meta):
    return ColumnTransformer([
        ('num',Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),meta['num_cols']),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),meta['cat_cols'])])


def pipe(meta, classifier):
    return Pipeline([('prep',prep(meta)),('clf',classifier)])


def candidates():
    return {
        'Logistic Regression':[LogisticRegression(C=c,max_iter=1000,random_state=SEED) for c in (.1,1.)],
        'Decision Tree':[DecisionTreeClassifier(max_depth=d,min_samples_leaf=30,random_state=SEED) for d in (5,9)],
        'Random Forest':[RandomForestClassifier(n_estimators=120,max_depth=d,min_samples_leaf=10,n_jobs=2,random_state=SEED) for d in (8,14)],
        'HistGradientBoosting':[HistGradientBoostingClassifier(max_iter=120,learning_rate=.06,max_leaf_nodes=l,random_state=SEED) for l in (15,31)]}


def split_at_freeze(df, freeze=FREEZE):
    # Chỉ dùng nhãn có đủ cửa sổ theo dõi trước/chính tại ngày đóng băng.
    observed = df.label_observed & df[TARGET].notna()
    mature = observed & df.label_observation_end.le(freeze) & df.snapshot_cutoff.lt(freeze)
    train = df[mature & df.cohort_year.le(2023)]
    val = df[mature & df.cohort_year.eq(2024)]
    test = df[observed & df.cohort_year.eq(2025) & df.snapshot_cutoff.ge(freeze)]
    for name, part in [('train',train),('validation',val),('test',test)]:
        require(part[TARGET].nunique()==2, f'Tập {name} thiếu một lớp nhãn')
    require(not set(train.record_uid)&set(val.record_uid) and not set(train.record_uid)&set(test.record_uid) and not set(val.record_uid)&set(test.record_uid),'Trùng sinh viên giữa tập')
    return train,val,test


def evaluate(y,p,threshold):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    if len(y)==0:
        return {'n':0}
    pred=p>=threshold
    k=max(1,int(np.ceil(.1*len(y))))
    # Ổn định khi xác suất bằng nhau; thứ tự đầu vào đã được cố định bằng khóa.
    top=np.argsort(-p,kind='stable')[:k]
    total=int(y.sum()); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()
    return {'n':len(y),'positive':total,'baseline':float(y.mean()),
            'average_precision':float(average_precision_score(y,p)) if total else None,
            'roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'brier':float(brier_score_loss(y,p)),'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)), 'f1':float(f1_score(y,pred,zero_division=0)),
            'recall_top10':float(y[top].sum()/total) if total else None,
            'alert_rate':float(pred.mean()),'confusion_matrix':[[tn,fp],[fn,tp]]}


def choose_threshold(y,p):
    # Mục tiêu F2 ưu tiên recall, giới hạn danh sách can thiệp <= 30% validation.
    options=[]
    for t in np.arange(.02,.951,.01):
        pred=p>=t
        if pred.mean()>.30 or not pred.any():
            continue
        pr=precision_score(y,pred,zero_division=0); re=recall_score(y,pred,zero_division=0)
        score=5*pr*re/(4*pr+re) if pr+re else 0.
        options.append((score,re,float(t)))
    require(bool(options),'Không tìm được ngưỡng hợp lệ trong ngân sách cảnh báo')
    return round(max(options)[2],4)


def calibration(y,p):
    table=pd.DataFrame({'y':np.asarray(y,dtype=int),'p':p}); table['bin']=pd.cut(table.p,np.linspace(0,1,11),include_lowest=True)
    return [{'bin':str(k),'n':int(len(g)),'xac_suat_trung_binh':float(g.p.mean()),'ty_le_thuc_te':float(g.y.mean())} for k,g in table.groupby('bin',observed=True)]


def fairness(frame,p,threshold,students):
    f=frame[KEY+[TARGET,'cohort_year','major_code','degree_type']].copy();f['p']=p
    f=f.merge(students[['record_uid','gender']],on='record_uid',validate='many_to_one')
    return {c:{str(k):evaluate(g[TARGET],g.p,threshold) if len(g)>=30 and g[TARGET].nunique()==2 else {'n':len(g),'note':'Mẫu/nhãn quá ít'} for k,g in f.groupby(c,dropna=False)} for c in ['gender','cohort_year','major_code','degree_type']}


def rolling_origin(df,meta):
    results=[]
    for origin, cohort, end in [('2025-02-03',2024,'2025-07-15'),('2026-02-03',2025,'2026-07-15'),('2027-02-03',2025,'2027-07-15')]:
        known=df.label_observed & df[TARGET].notna()
        train=df[known & df.cohort_year.lt(cohort) & df.label_observation_end.lt(origin) & df.snapshot_cutoff.lt(origin)]
        evaluation=df[known & df.cohort_year.eq(cohort) & df.snapshot_cutoff.between(origin,end)]
        if train[TARGET].nunique()!=2 or evaluation[TARGET].nunique()!=2:
            results.append({'origin':origin,'note':'Không đủ hai lớp nhãn'});continue
        models={'Logistic Regression':LogisticRegression(C=1,max_iter=1000,random_state=SEED),
                'Ensemble cố định':VotingClassifier(estimators=[('rf',RandomForestClassifier(n_estimators=80,max_depth=8,min_samples_leaf=10,random_state=SEED,n_jobs=2)),('hgb',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,random_state=SEED))],voting='soft',weights=[.5,.5])}
        metrics={}
        for name, clf in models.items():
            m=pipe(meta,clf).fit(model_inputs(train,meta),train[TARGET].astype(int))
            p=m.predict_proba(model_inputs(evaluation,meta))[:,1]
            metrics[name]=evaluate(evaluation[TARGET],p,.21)
        results.append({'origin':origin,'train_n':len(train),'train_horizon_max':train.label_observation_end.max(),'test_cutoff_min':evaluation.snapshot_cutoff.min(),'nguong_co_dinh':.21,'metrics':metrics})
    return results


def train_models(df,meta,freeze=FREEZE):
    train,val,test=split_at_freeze(df,freeze)
    # Chỉ công bố dự báo trong các kỳ có đủ hai lớp và >= 100 snapshot huấn luyện.
    counts=train.groupby('snapshot_term_index')[TARGET].agg(['size','sum'])
    supported=counts.index[(counts['size']>=100)&(counts['sum']>=5)&(counts['size']-counts['sum']>=5)].astype(int).tolist()
    require(bool(supported),'Không có học kỳ đủ dữ liệu huấn luyện')
    train=train[train.snapshot_term_index.isin(supported)];val=val[val.snapshot_term_index.isin(supported)];test=test[test.snapshot_term_index.isin(supported)]
    for part in (train,val,test):require(part[TARGET].nunique()==2,'Thiếu lớp nhãn trong phạm vi model')
    report={'pham_vi_hoc_ky':supported,'moc_dong_bang':freeze,'so_luong_tap':{'train':len(train),'validation':len(val),'test':len(test)},'train_horizon_max':train.label_observation_end.max(),'validation_horizon_max':val.label_observation_end.max(),'test_cutoff_min':test.snapshot_cutoff.min()}
    # Chọn mô hình không dùng tín hiệu hành chính để đánh giá cảnh báo sớm; so sánh ablation cùng thuật toán.
    reduced=subset_meta(meta,[c for c in meta['feature_cols'] if c!=ADMIN])
    Xtr=model_inputs(train,reduced);Xv=model_inputs(val,reduced);Xt=model_inputs(test,reduced)
    yt=train[TARGET].astype(int);yv=val[TARGET].astype(int);ytest=test[TARGET].astype(int)
    selected={}; params={};vp={}
    for name,options in candidates().items():
        print(f"Huấn luyện và chọn tham số: {name}",flush=True)
        scored=[]
        for clf in options:
            m=pipe(reduced,clf).fit(Xtr,yt);p=m.predict_proba(Xv)[:,1]
            scored.append((average_precision_score(yv,p),m,clf,p))
        _,m,clf,p=max(scored,key=lambda z:z[0]);selected[name]=m;params[name]=clf.get_params();vp[name]=p
    print("Tối ưu trọng số tổ hợp trên tập validation",flush=True)
    weights=max(np.arange(0,1.01,.1),key=lambda w:average_precision_score(yv,w*vp['Random Forest']+(1-w)*vp['HistGradientBoosting']))
    m=pipe(reduced,VotingClassifier(estimators=[('rf',clone(selected['Random Forest'].named_steps['clf'])),('hgb',clone(selected['HistGradientBoosting'].named_steps['clf']))],voting='soft',weights=[float(weights),float(1-weights)])).fit(Xtr,yt)
    selected['Ensemble']=m;vp['Ensemble']=m.predict_proba(Xv)[:,1]
    # Cố định lựa chọn trước khi dùng test. Chọn theo AP validation, không bắt buộc tổ hợp phải thắng.
    winner=max(selected,key=lambda name:average_precision_score(yv,vp[name]));model=selected[winner]
    threshold=choose_threshold(yv,vp[winner]);prob=model.predict_proba(Xt)[:,1]
    report.update({'model_chon':winner,'trong_so_rf':float(weights),'tham_so':params,'nguong':threshold,
                   'benchmark':{name:{'validation':evaluate(yv,vp[name],threshold),'test':evaluate(ytest,m.predict_proba(Xt)[:,1],threshold)} for name,m in selected.items()}})
    early=~test[ADMIN].astype(bool);report['test_early']=evaluate(ytest[early],prob[early],threshold)
    report['test_by_term']={str(int(term)):evaluate(ytest[mask],prob[mask],threshold) for term in sorted(test.snapshot_term_index.unique()) for mask in [test.snapshot_term_index.eq(term)]}
    full=pipe(meta,clone(model.named_steps['clf'])).fit(model_inputs(train,meta),yt)
    pfull=full.predict_proba(model_inputs(test,meta))[:,1];report['ablation']={'co_hanh_chinh':evaluate(ytest,pfull,threshold),'khong_hanh_chinh':evaluate(ytest,prob,threshold),'co_hanh_chinh_early':evaluate(ytest[early],pfull[early],threshold)}
    report['calibration']=calibration(ytest,prob)
    # Giải thích toàn cục trên validation, không diễn giải trọng số là nguyên nhân.
    sample=val.sample(min(1500,len(val)),random_state=SEED)
    pi=permutation_importance(model,model_inputs(sample,reduced),sample[TARGET].astype(int),scoring='average_precision',n_repeats=2,random_state=SEED,n_jobs=1)
    report['feature_importance']=[{'feature':c,'ap_decrease':float(v),'std':float(sd)} for c,v,sd in sorted(zip(reduced['feature_cols'],pi.importances_mean,pi.importances_std),key=lambda x:-x[1])]
    print("Đánh giá theo thời gian và giải thích đặc trưng",flush=True)
    report['rolling_origin']=rolling_origin(df,reduced)
    best=report['benchmark'][winner]; tm=best['test']
    reduced.update(model_available_as_of=freeze,supported_terms=supported,threshold=threshold,model_name=winner,
                   val_pr_auc=best['validation']['average_precision'],test_pr_auc=tm['average_precision'],test_roc_auc=tm['roc_auc'],test_brier=tm['brier'],
                   test_precision=tm['precision'],test_recall=tm['recall'],test_f1=tm['f1'],test_baseline=tm['baseline'],test_recall_top10=tm['recall_top10'],global_feature_importance=report['feature_importance'][:10])
    return model,reduced,report,test,prob


class StressDataset(Dataset):
    def __init__(self,run):
        self.run=Path(run); self.manifest=json.loads((self.run/'manifests/generation_manifest.json').read_text(encoding='utf-8'));self.cache={}
        require(self.manifest.get('scenario_id')=='V3_2_STRESS' and self.manifest.get('research_revision')=='cmc-policy-2026-01' and self.manifest.get('n_students')==5000,'Tập STRESS không hợp lệ')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='Thư mục phiên bản mới bên trong data, chưa tồn tại')
    args=p.parse_args(); output=args.output.resolve()
    require(output.is_relative_to((ROOT/'data').resolve()) and not output.exists(),'Chỉ ghi vào thư mục mới bên trong data/')
    source=Dataset();require(source.manifest.get('research_revision')=='cmc-policy-2026-01','Phải sinh dữ liệu hiệu chỉnh trước khi train')
    require(audit(source,None,ROOT/'data/kiem_tra_truoc_huan_luyen'),'Dữ liệu hiệu chỉnh chưa đạt, dừng huấn luyện')
    df=join_data(source);meta=feature_meta(source,df)
    model,meta,report,test,prob=train_models(df,meta)
    report['fairness']=fairness(test,prob,meta['threshold'],source.table('students'))
    stress=StressDataset(source.run.with_name('V3_2_STRESS'));st=join_data(stress)
    require(not set(source.table('students').student_id)&set(stress.table('students').student_id),'Trùng mã sinh viên MAIN/STRESS')
    st=st[st.label_observed & st.snapshot_cutoff.ge(FREEZE) & st.snapshot_term_index.isin(meta['supported_terms'])]
    require(st[TARGET].nunique()==2,'STRESS thiếu hai lớp nhãn')
    sp=model.predict_proba(model_inputs(st,meta))[:,1];report['stress']=evaluate(st[TARGET],sp,meta['threshold']);report['stress_fairness']=fairness(st,sp,meta['threshold'],stress.table('students'))
    scored=df.copy();scored['risk_probability']=model.predict_proba(model_inputs(scored,meta))[:,1];scored['risk_level']=[risk_level(v,meta['threshold']) for v in scored.risk_probability]
    students=source.table('students');static=['record_uid','class_code','major_name']
    scored=scored.merge(students[static],on='record_uid',validate='many_to_one')
    meta.update(dataset_manifest_sha256=digest(source.run/'manifests/generation_manifest.json'),version=output.name,sklearn_version=sklearn_version)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='kiemit_') as tmp:
        stage=Path(tmp)/'model';stage.mkdir()
        joblib.dump(model,stage/'dropout_pipeline.joblib')
        scored.to_csv(stage/'student_snapshots_scored.csv.gz',index=False,compression='gzip')
        (stage/'model_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        # Kiểm tra sau khi nạp lại, trước khi đổi tên thư mục kết quả.
        restored=joblib.load(stage/'dropout_pipeline.joblib')
        persisted=pd.read_csv(stage/'student_snapshots_scored.csv.gz',low_memory=False,float_precision='round_trip')
        sample=persisted.sample(min(512,len(persisted)),random_state=SEED)
        require(np.allclose(restored.predict_proba(model_inputs(sample,meta))[:,1],sample.risk_probability,atol=1e-10,rtol=0),'Sai xác suất khi đọc lại model')
        rt,_=runtime_view(persisted,source,'DEMO');ac,latest=academic_view(rt,source,'DEMO')
        from core.prediction import apply_prediction_scope
        ac=apply_prediction_scope(ac,meta);latest=apply_prediction_scope(latest,meta)
        summary=build_analytics_summary(ac,visible_attempts(source.table('course_attempts'),latest),meta['threshold'],len(rt))
        (stage/'analytics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        (stage/'bao_cao_thuc_nghiem.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        del persisted,rt,ac,latest
        require(audit(source,stage/'student_snapshots_scored.csv.gz',stage/'kiem_tra_du_lieu'),'Không đạt kiểm tra dữ liệu; không công bố model')
        # Dấu kiểm tra toàn vẹn chỉ tạo sau kiểm tra thành công.
        files=['dropout_pipeline.joblib','student_snapshots_scored.csv.gz','model_metadata.json','analytics_summary.json','bao_cao_thuc_nghiem.json']
        (stage/'manifest.json').write_text(json.dumps({'files':{f:digest(stage/f) for f in files},'dataset_manifest_sha256':meta['dataset_manifest_sha256']},indent=2),encoding='utf-8')
        stage.rename(output)
    print(f'Đã kiểm tra và xuất phiên bản {output}. Chưa thay thế mô hình cũ.')
    print(json.dumps(report['benchmark'][meta['model_name']]['test'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
