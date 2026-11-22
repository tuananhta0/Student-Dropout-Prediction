"""Đóng gói riêng dữ liệu nghiên cứu và gói chạy đã nghiệm thu; không chứa tài khoản/API key."""
import argparse
import json
from pathlib import Path
import tempfile
import zipfile
from core.data_pipeline import ROOT, MAIN, Dataset, digest, require
from scripts.train_model import StressDataset
from scripts.release_model import verify_pair, MODEL_FILES


def pack_research(source_archive,dataset,model,output):
    source_archive,dataset,model,output=map(Path,(source_archive,dataset,model,output))
    require(not output.exists(),'Gói đã tồn tại; không ghi đè')
    meta=verify_pair(dataset,model);main=Dataset(dataset/MAIN);stress=StressDataset(main.run.with_name('V3_2_STRESS'))
    require(digest(source_archive)==main.manifest['parent_archive_sha256']==stress.manifest['parent_archive_sha256'],'Archive nguồn không khớp dữ liệu tái sinh')
    files={'nguon/dataset-goc.zip':source_archive}
    for ds in (main,stress):
        for name in ds.manifest['table_artifacts']:
            info=ds.manifest['table_artifacts'][name]
            # Toàn bộ bảng do manifest sinh ra (gồm weekly/support), không quét thư mục data ngoài run.
            rel=info.get('path') or info.get('relative_path') or info.get('file')
            candidates=[ds.run/rel] if rel and (ds.run/rel).is_file() else list(ds.run.rglob(Path(rel).name if rel else name+'.csv.gz'))
            # Các split cùng tên ở hai view phải được xác định bằng hash, không chọn tùy ý.
            matched=[path for path in candidates if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ds.run.resolve()) and digest(path)==info['sha256']]
            require(bool(matched),f'Sai hoặc thiếu bảng {name}')
            for path in matched:
                files[f'du_lieu/{ds.run.name}/{path.relative_to(ds.run).as_posix()}']=path
        # Bao gồm tài liệu và cấu hình của bộ sinh, không đưa nội dung ngoài run vào gói.
        for folder in ('docs','manifests','reports'):
            for path in (ds.run/folder).rglob('*'):
                if path.is_file() and not path.is_symlink(): files[f'du_lieu/{ds.run.name}/{path.relative_to(ds.run).as_posix()}']=path
    for name in MODEL_FILES:files['model/'+name]=model/name
    for rel in ('kiem_tra_ml/bao_cao.json','kiem_tra_rag/bao_cao.json','kiem_tra_rag/bao_cao_mo_rong.json','kiem_tra_du_lieu/kiem_tra_du_lieu.json','kiem_tra_stress/kiem_tra_du_lieu.json'):
        path=ROOT/'data'/rel
        if path.is_file():files['ket_qua/'+rel]=path
    manifest={'version':meta['version'],'files':{k:{'sha256':digest(v),'bytes':v.stat().st_size} for k,v in files.items()}}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staged=Path(tmp)/'research.zip'
        with zipfile.ZipFile(staged,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for k,v in files.items():z.write(v,k)
            z.writestr('research_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        with zipfile.ZipFile(staged) as z: require(z.testzip() is None,'Gói ZIP lỗi CRC')
        staged.rename(output)
    print('Gói nghiên cứu:',output,'SHA256:',digest(output))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source_archive',type=Path);p.add_argument('--dataset',type=Path,default=ROOT/'data/du_lieu_hieu_chinh');p.add_argument('--model',type=Path,default=ROOT/'data/models/cmc-research-v1');p.add_argument('--output',type=Path,default=ROOT/'data/cmc-research-full.zip');a=p.parse_args();pack_research(a.source_archive,a.dataset,a.model,a.output)
if __name__=='__main__':main()
