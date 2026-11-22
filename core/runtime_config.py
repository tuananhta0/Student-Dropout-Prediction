"""Chọn cặp dữ liệu - model chính thức, không tự lùi về baseline."""
import os
import json
from pathlib import Path
from core.data_pipeline import ROOT, MAIN, digest


def configure_runtime():
    dataset=os.environ.get('CMC_DATASET_ROOT')
    model=os.environ.get('CMC_MODEL_DIR')
    if bool(dataset)!=bool(model):
        raise RuntimeError('Phải cấu hình đồng thời CMC_DATASET_ROOT và CMC_MODEL_DIR.')
    if not dataset:
        if os.environ.get('CMC_ALLOW_BASELINE')=='1':
            dataset,model=ROOT/'dataset_full',ROOT/'artifacts_model'
        else:
            release=ROOT/'data/releases/cmc-research-v1'
            if release.exists():
                dataset,model=release/'dataset',release/'model'
            else:
                dataset,model=ROOT/'data/du_lieu_hieu_chinh',ROOT/'data/models/cmc-research-v1'
    dataset,model=Path(dataset).resolve(),Path(model).resolve()
    if not (dataset/MAIN/'manifests/generation_manifest.json').is_file():
        raise RuntimeError('Thiếu dữ liệu tương ứng. Xem docs/ban_giao.md.')
    if not (model/'manifest.json').is_file() and os.environ.get('CMC_ALLOW_BASELINE')!='1':
        raise RuntimeError('Thiếu manifest mô hình nghiên cứu. Cài gói bàn giao; không tự dùng baseline.')
    if os.environ.get('CMC_ALLOW_BASELINE')!='1':
        meta=json.loads((model/'model_metadata.json').read_text(encoding='utf-8'))
        manifest=json.loads((model/'manifest.json').read_text(encoding='utf-8'))
        actual=digest(dataset/MAIN/'manifests/generation_manifest.json')
        if meta.get('version')!='cmc-research-v1' or meta.get('dataset_manifest_sha256')!=actual or manifest.get('dataset_manifest_sha256')!=actual:
            raise RuntimeError('Cặp dữ liệu và model cmc-research-v1 không khớp.')
    os.environ['CMC_DATASET_ROOT']=str(dataset)
    os.environ['CMC_MODEL_DIR']=str(model)
    return str(model)
