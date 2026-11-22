"""Đóng gói/cài đặt cặp dữ liệu - mô hình đã huấn luyện; không huấn luyện lại."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from core.data_pipeline import ROOT, MAIN, TABLES, Dataset, digest, require

MODEL_FILES=('dropout_pipeline.joblib','model_metadata.json','student_snapshots_scored.csv.gz','analytics_summary.json','bao_cao_thuc_nghiem.json','manifest.json')
DATA_FILES=('manifests/generation_manifest.json', *(v[0] for v in TABLES.values()))
VERSION='cmc-research-v1'


def verify_pair(dataset, model):
    ds=Dataset(dataset/MAIN)
    require(ds.manifest.get('research_revision')=='cmc-policy-2026-01','Chỉ đóng gói dữ liệu đã hiệu chỉnh')
    for name in TABLES: ds.table(name)
    meta=json.loads((model/'model_metadata.json').read_text(encoding='utf-8'))
    mf=json.loads((model/'manifest.json').read_text(encoding='utf-8'))
    sha=digest(ds.run/'manifests/generation_manifest.json')
    require(meta.get('version')==VERSION and meta.get('dataset_manifest_sha256')==sha and mf.get('dataset_manifest_sha256')==sha,'Model và dữ liệu không cùng phiên bản')
    for name in MODEL_FILES[:-1]: require(digest(model/name)==mf['files'].get(name),f'Sai SHA256: {name}')
    return meta


def pack(dataset,model,output):
    dataset,model,output=map(Path,(dataset,model,output))
    require(not output.exists(),'Gói đã tồn tại; chọn tên khác')
    meta=verify_pair(dataset,model)
    files={f'dataset/{MAIN.as_posix()}/{rel}':dataset/MAIN/rel for rel in DATA_FILES}
    files.update({f'model/{name}':model/name for name in MODEL_FILES})
    for f in (model/'kiem_tra_du_lieu').glob('*.json'):
        if f.is_file(): files[f'model/kiem_tra_du_lieu/{f.name}']=f
    manifest={'version':VERSION,'dataset_manifest_sha256':meta['dataset_manifest_sha256'],'files':{k:{'sha256':digest(v),'bytes':v.stat().st_size} for k,v in files.items()}}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        out=Path(tmp)/output.name
        with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for k,v in files.items(): z.write(v,k)
            z.writestr('release_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        out.rename(output)
    print('Đã tạo gói:',output)
    print('SHA256 (chia sẻ riêng với nhóm):',digest(output))


def install(archive, destination, expected_sha):
    archive,destination=map(Path,(archive,destination))
    require(not destination.exists(),'Thư mục đã tồn tại; không ghi đè')
    require(digest(archive)==expected_sha.lower(),'SHA256 gói bàn giao không khớp')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        names=z.namelist(); require(len(set(names))==len(names),'Tên file trùng trong gói')
        manifest=json.loads(z.read('release_manifest.json'))
        entries=manifest['files']
        required={f'dataset/{MAIN.as_posix()}/{r}' for r in DATA_FILES}|{f'model/{f}' for f in MODEL_FILES}
        require(manifest['version']==VERSION and required<=set(entries) and set(names)==set(entries)|{'release_manifest.json'},'Gói không đúng cấu trúc')
        stage=Path(tmp)/'release';stage.mkdir()
        for name,info in entries.items():
            path=PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name and (name in required or (len(path.parts)==3 and path.parts[:2]==('model','kiem_tra_du_lieu') and path.suffix=='.json')),'Đường dẫn gói không hợp lệ')
            require(z.getinfo(name).file_size==info['bytes'],'Kích thước không khớp')
            target=stage.joinpath(*path.parts);target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(name) as src,target.open('wb') as out: shutil.copyfileobj(src,out)
            require(digest(target)==info['sha256'],f'Sai SHA256: {name}')
        verify_pair(stage/'dataset',stage/'model')
        (stage/'release_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        stage.rename(destination)
    print('Đã cài và kiểm tra gói:',destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('pack');p.add_argument('--dataset',type=Path,default=ROOT/'data/du_lieu_hieu_chinh');p.add_argument('--model',type=Path,default=ROOT/'data/models'/VERSION);p.add_argument('--output',type=Path,default=ROOT/'data'/f'{VERSION}.zip')
    i=sub.add_parser('install');i.add_argument('archive',type=Path);i.add_argument('--sha256',required=True);i.add_argument('--destination',type=Path,default=ROOT/'data/releases'/VERSION)
    a=parser.parse_args()
    if a.action=='pack': pack(a.dataset,a.model,a.output)
    else: install(a.archive,a.destination,a.sha256)

if __name__=='__main__':main()
