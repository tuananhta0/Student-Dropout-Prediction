"""Tái sinh dữ liệu mô phỏng bằng bộ sinh V3.2.1 đã kiểm tra, không ghi đè bản gốc."""
import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from core.data_pipeline import ROOT, MAIN, Dataset, TABLES, admission_evidence, digest

ARCHIVE_SHA256 = 'd5923367a3907d0edd73f7cb37105e8913477a1b9f57792daf148e4f871fd6a4'
PREFIX = 'CMC_Synthetic_Dataset_V3.2.1/'
REVISION = 'cmc-policy-2026-01'


def replace(path, old, new):
    text = path.read_text(encoding='utf-8')
    if text.count(old) != 1:
        raise ValueError(f'Mã nguồn bộ sinh không khớp: {path.name}: {old[:60]}')
    path.write_text(text.replace(old, new), encoding='utf-8')


def prepare_generator(archive, workspace):
    if digest(Path(archive)) != ARCHIVE_SHA256:
        raise ValueError('Archive khác bản V3.2.1 đã đối chiếu SHA256; không chạy mã nguồn chưa kiểm tra.')
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.startswith(PREFIX) or name.endswith('/'):
                continue
            rel = Path(name[len(PREFIX):])
            if '..' in rel.parts or rel.is_absolute():
                raise ValueError('Đường dẫn archive không hợp lệ')
            if rel.parts[0] not in ('src', 'config') and rel.as_posix() != 'references/source_snapshots_v22.zip':
                continue
            path = workspace / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(name))
    src = workspace / 'src'
    # Phân loại sau khi làm tròn half-up đến một chữ số thập phân.
    replace(src/'rules.py', '    if not exam_eligible or final_exam10 <= 2 or score10 < 4:',
            '    from decimal import Decimal, ROUND_HALF_UP\n    score10 = float(Decimal(str(score10)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))\n    if not exam_eligible or final_exam10 <= 2 or score10 < 4:')
    # Không giới hạn hai môn học lại khi không đủ tải dù còn nhiều môn nợ.
    replace(src/'registration_planner.py', '    available=sum(c[1] for c in catalog if c[0] not in passed)',
            '''    if credits < minimum:
        for cid in sorted(fails, key=lambda c: (-fails[c], c)):
            if cid not in selected and cid not in passed and credits + byid[cid][1] <= maximum:
                selected.append(cid)
                credits += byid[cid][1]
            if credits >= minimum:
                break
    available=sum(c[1] for c in catalog if c[0] not in passed)''')
    # Điều kiện thi dựa trên đúng độ chính xác của số giờ được lưu.
    replace(src/'generate.py', '                eligible=bool(abs_unexc<=0.20 and abs_total<=0.30)',
            '''                unexcused_hours=round(abs_unexc*credit*15,4)
                excused_hours=round((abs_total-abs_unexc)*credit*15,4)
                u=int(round(unexcused_hours*10000)); e=int(round(excused_hours*10000)); h=credit*15*10000
                eligible=bool(5*u<=h and 10*(u+e)<=3*h)''')
    replace(src/'generate.py', "'excused_absent_hours':round((abs_total-abs_unexc)*credit*15,4),\n                   'unexcused_absent_hours':round(abs_unexc*credit*15,4)",
            "'excused_absent_hours':excused_hours,\n                   'unexcused_absent_hours':unexcused_hours")
    # Ghi theo lô, không giữ hàng triệu bản ghi latent trong RAM.
    replace(src/'generate.py', "'graduation_clearance':clearance_rows}",
            "'graduation_clearance':clearance_rows,'weekly_state_hidden':hidden_rows,\n       'synthetic_shock_events':shock_rows}")
    replace(src/'generate.py', '        if (i+1)%500==0:flush()', '        if (i+1)%500==0:\n            flush()\n            print(f"Đã sinh {i+1}/{n} sinh viên", flush=True)')
    return workspace / 'references/source_snapshots_v22.zip'


def run_generator(workspace, source_zip, config, run):
    cfg = dict(config)
    cfg['snapshot_terms'] = list(range(1, int(cfg['max_simulation_terms']) + 1))
    path = workspace/'config/runtime_research.yaml'
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'src.generate', '--config', str(path), '--source', str(source_zip), '--out', str(run)], cwd=workspace, check=True)
    manifest_path = run/'manifests/generation_manifest.json'
    # Điểm dưới ngưỡng có thể so sánh: chuyển sang thiếu có lý do, không nâng điểm.
    students_path = run/'master/students.csv.gz'
    stu = pd.read_csv(students_path, low_memory=False)
    evidence = admission_evidence(stu)
    below = evidence.admission_evidence_status.eq('BELOW_REVIEWED_FLOOR')
    stu['admission_score_missing_reason'] = ''
    stu.loc[stu.admission_score.isna(), 'admission_score_missing_reason'] = 'SYNTHETIC_SOURCE_MISSING'
    stu.loc[stu.admission_method.eq('CMC303_TuyenThang'), 'admission_score_missing_reason'] = 'NOT_APPLICABLE_DIRECT_ADMISSION'
    (run/'reports').mkdir(exist_ok=True)
    stu.loc[below, ['record_uid','student_id','cohort_year','major_code','admission_method','admission_score']].assign(
        published_floor=evidence.loc[below,'published_floor']).to_csv(run/'reports/admission_excluded.csv', index=False)
    stu.loc[below, ['admission_score','admission_score_normalized']] = float('nan')
    stu.loc[below,'admission_score_missing_reason'] = 'BELOW_REVIEWED_FLOOR_EXCLUDED'
    stu.to_csv(students_path,index=False,compression='gzip',float_format='%.8f')
    # Điểm đầu vào không tham gia mô phỏng diễn biến; dựng lại feature và nhãn sau hiệu chỉnh.
    subprocess.run([sys.executable, '-c', '''import sys,json,yaml
from pathlib import Path
from src.snapshots import build_snapshots
run=Path(sys.argv[1]);cfg=yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"));p=run/"manifests/generation_manifest.json"
m=json.loads(p.read_text(encoding="utf-8"));m["table_artifacts"].update(build_snapshots(run,cfg));p.write_text(json.dumps(m,ensure_ascii=False),encoding="utf-8")''',str(run),str(path)],cwd=workspace,check=True)
    subprocess.run([sys.executable, '-m', 'src.finalize', '--run', str(run), '--source', str(source_zip)], cwd=workspace,check=True)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest.update(research_revision=REVISION, parent_archive_sha256=ARCHIVE_SHA256,
                    student_id_format='BxxYYNNNN', snapshot_terms=cfg['snapshot_terms'],
                    admissions_below_floor_excluded=int(below.sum()),
                    generator_source_sha256={p.name:digest(p) for p in sorted((workspace/'src').glob('*.py'))})
    manifest['known_limitations'] += ['Lịch sử, nhãn và điểm được tái sinh; không giữ nguyên từng sự kiện cũ.',
        'Điểm tuyển sinh thiếu giữ nguyên; điểm dưới ngưỡng đối chiếu được loại khỏi đặc trưng, không khẳng định quyết định tuyển sinh.',
        'Khối lượng còn lại cuối chương trình là ngoại lệ giả định mô phỏng.']
    (run/'manifests/generator_config.yaml').write_bytes(path.read_bytes())
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--destination', type=Path, default=ROOT/'data/du_lieu_hieu_chinh')
    p.add_argument('--stress', action='store_true', help='Sinh thêm tập kiểm thử STRESS riêng biệt')
    args = p.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f'Không ghi đè: {destination}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent,prefix='sinh_du_lieu_') as tmp:
        tmp = Path(tmp)
        workspace = tmp/'generator'; workspace.mkdir()
        source_zip = prepare_generator(args.archive,workspace)
        target = tmp/'dataset'
        configs = [('v3_2_1_base_frozen.yaml',MAIN)]
        if args.stress:
            configs.append(('v3_2_1_stress.yaml',MAIN.with_name('V3_2_STRESS')))
        for config, rel in configs:
            cfg = yaml.safe_load((workspace/'config'/config).read_text(encoding='utf-8'))
            print(f'Đang tái sinh {cfg["scenario_id"]} ({cfg["n_students"]} sinh viên)',flush=True)
            m=run_generator(workspace,source_zip,cfg,target/rel)
            # Đảm bảo dữ liệu tĩnh từ lần sinh mới không đổi danh tính mô phỏng.
            with zipfile.ZipFile(args.archive) as z:
                with z.open(PREFIX+rel.as_posix()+'/master/students.csv.gz') as stream:
                    old=pd.read_csv(stream,compression='gzip',usecols=['record_uid','student_id','curriculum_id'])
                new=pd.read_csv(target/rel/'master/students.csv.gz',usecols=['record_uid','student_id','curriculum_id'])
                cols=['record_uid','student_id','curriculum_id']
                if not old[cols].sort_values('record_uid').reset_index(drop=True).equals(new[cols].sort_values('record_uid').reset_index(drop=True)):
                    raise ValueError('Danh tính hoặc chương trình đào tạo không còn khớp nguồn; dừng công bố bộ sinh mới')
                old_manifest=json.loads(z.read(PREFIX+rel.as_posix()+'/manifests/generation_manifest.json'))
                report={'so_sinh_vien_khop':len(new),'bang_nguon':{k:v['rows'] for k,v in old_manifest['table_artifacts'].items()},
                        'bang_moi':{k:v['rows'] for k,v in m['table_artifacts'].items()},'diem_tuyen_sinh_loai_khoi_feature':m['admissions_below_floor_excluded']}
                (target/rel/'reports/so_sanh_nguon.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        ds = Dataset(target/MAIN)
        for name in TABLES:
            ds.table(name)
        target.rename(destination)
    print(f'Đã sinh dữ liệu nghiên cứu mới: {destination}. Bản gốc không đổi.')

if __name__ == '__main__':
    main()
