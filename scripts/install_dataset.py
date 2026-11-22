"""Cài bảng MAIN cần thiết, không thay đổi dữ liệu nguồn."""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from core.data_pipeline import Dataset, MAIN, ROOT, TABLES, digest, main_run


def supplement(archive, destination):
    """Bổ sung bảng xét tốt nghiệp còn thiếu, không ghi đè dữ liệu đã cài."""
    run = Path(destination).resolve() / MAIN
    ds = Dataset(run)
    rel = TABLES['graduation_clearance'][0]
    for name in TABLES:
        if name != 'graduation_clearance':
            ds.table(name)
    target = run / rel
    if target.exists():
        ds.table('graduation_clearance')
        print('Bảng xét tốt nghiệp đã có và hợp lệ.')
        return
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        suffix = '/' + MAIN.as_posix() + '/manifests/generation_manifest.json'
        matches = [n for n in z.namelist() if n.endswith(suffix)]
        if len(matches) != 1 or z.read(matches[0]) != (run/'manifests/generation_manifest.json').read_bytes():
            raise ValueError('Manifest không khớp dữ liệu đã cài')
        prefix = matches[0][:-len('manifests/generation_manifest.json')]
        staged = Path(tmp)/target.name
        with z.open(prefix+rel) as src, staged.open('wb') as out:
            shutil.copyfileobj(src,out)
        if digest(staged) != ds.manifest['table_artifacts']['graduation_clearance']['sha256']:
            raise ValueError('Bảng xét tốt nghiệp không khớp SHA256')
        staged.rename(target)
    ds.table('graduation_clearance')
    print('Đã bổ sung và kiểm tra bảng xét tốt nghiệp.')


def install(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'{destination} đã tồn tại; dùng --supplement hoặc chọn thư mục khác')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp, zipfile.ZipFile(archive) as z:
        target = Path(tmp) / 'dataset'
        run = target / MAIN
        suffix = '/' + MAIN.as_posix() + '/manifests/generation_manifest.json'
        matches = [n for n in z.namelist() if n.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError('Cần đúng một manifest V3_2_MAIN')
        prefix = matches[0][:-len('manifests/generation_manifest.json')]
        for rel in ['manifests/generation_manifest.json', *(v[0] for v in TABLES.values())]:
            path = run / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(prefix + rel) as src, path.open('wb') as out:
                shutil.copyfileobj(src, out)
        ds = Dataset(run)
        for name in TABLES:
            ds.table(name)
        target.rename(destination)
    print(f'Đã cài và kiểm tra dữ liệu MAIN: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--destination', type=Path, default=main_run().parents[2])
    parser.add_argument("--supplement", action="store_true", help="Bổ sung bảng còn thiếu vào dữ liệu hiện có")
    args = parser.parse_args()
    (supplement if args.supplement else install)(args.archive, args.destination)
