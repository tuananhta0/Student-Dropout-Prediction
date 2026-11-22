"""Install only the MAIN tables needed for runtime and audit; no source changes."""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from data_pipeline import Dataset, MAIN, ROOT, TABLES


def install(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'{destination} already exists; choose a new folder or rename existing data first')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp, zipfile.ZipFile(archive) as z:
        target = Path(tmp) / 'dataset'
        run = target / MAIN
        suffix = '/' + MAIN.as_posix() + '/manifests/generation_manifest.json'
        matches = [n for n in z.namelist() if n.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError('Expected exactly one V3_2_MAIN manifest')
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
    print(f'Installed verified MAIN dataset: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--destination', type=Path, default=ROOT / 'dataset_full')
    args = parser.parse_args()
    install(args.archive, args.destination)
