"""One acceptance runner for the final V3 candidate (Windows/Linux).

Start only owned localhost processes, never modify source or commit. The
external delivery installer commits only after this runner succeeds.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx
from scripts.report_provenance import provenance

ROOT=Path(__file__).resolve().parents[1]


def environment(cloud=False):
    env=dict(os.environ);env.update(CMC_DATA_MODE='DEMO',CMC_RAG_CLOUD='1' if cloud else '0',CMC_CHAT_AGENT='1' if cloud else '0',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    # Never let old development overrides select the historical baseline.
    env.pop('CMC_ALLOW_BASELINE',None)
    return env


@contextmanager
def server(out,cloud,port):
    # Bind probe first, refuse an existing server rather than killing it.
    with socket.socket() as probe:
        try:probe.bind(('127.0.0.1',port))
        except OSError as exc:raise RuntimeError(f'Port {port} occupied; close your running Uvicorn and rerun.') from exc
    url=f'http://127.0.0.1:{port}'
    label='cloud' if cloud else 'offline'
    with (out/(label+'_uvicorn.log')).open('w',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-B','-X','utf8','-m','uvicorn','app:app','--host','127.0.0.1','--port',str(port),'--no-proxy-headers','--no-use-colors'],cwd=ROOT,env=environment(cloud),stdout=log,stderr=subprocess.STDOUT)
        try:
            until=time.monotonic()+150
            with httpx.Client(trust_env=False,timeout=2) as client:
                while time.monotonic()<until:
                    if child.poll() is not None:raise RuntimeError(f'{label} Uvicorn exited; see {log.name}')
                    try:
                        r=client.get(url+'/')
                        if r.status_code==200 and 'portal.js' in r.text:break
                    except httpx.HTTPError:pass
                    time.sleep(.5)
                else:raise RuntimeError(f'{label} Uvicorn startup timeout; see {log.name}')
            print(f'{label} localhost HTTP ready at {url}',flush=True)
            yield url
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/kiem_tra_rag/v3_final');p.add_argument('--port',type=int,default=8000);p.add_argument('--resume',action='store_true');p.add_argument('--bank',type=Path);p.add_argument('--audit',type=Path);a=p.parse_args()
    if not 1024<=a.port<=65534: p.error('port must be 1024..65534')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);status_path=out/'run_status.json'
    source=provenance();fingerprint={k:source[k] for k in ('source_commit','source_tree','source_dirty','tracked_diff_sha256')}
    stages={name:'NOT_RUN' for name in ('python','javascript','regression','runtime','academic','security','api_1800','http','browser','cloud_agent')}
    if a.resume and status_path.exists():
        prev=json.loads(status_path.read_text(encoding='utf-8-sig'))
        if prev.get('source')!=fingerprint:raise RuntimeError('Cannot reuse reports for another source tree')
        stages.update({k:v for k,v in prev.get('stages',{}).items() if k in stages and v in ('PASS','COMPLETE_NOT_SEMANTIC_PASS','OPTIONAL_NOT_RUN')})
    def save():status_path.write_text(json.dumps({'source':fingerprint,'stages':stages,'semantic_review':'DEVELOPER_MANUAL_OPTIONAL','provenance':provenance()},ensure_ascii=False,indent=2),encoding='utf-8')
    def command(name,argv,cloud=False):
        if stages[name]=='PASS':return
        stages[name]='RUNNING';save();print('Running '+name,flush=True)
        try:subprocess.run(argv,cwd=ROOT,env=environment(cloud),check=True)
        except BaseException:stages[name]='FAIL';save();raise
        stages[name]='PASS';save()
    def pymod(module,*args):return [sys.executable,'-B','-X','utf8','-m',module,*map(str,args)]
    save()
    try:
        command('python',pymod('unittest','discover','-s','tests','-v'))
        command('javascript',['node','--test','tests/portal_async.test.cjs'])
        command('regression',pymod('scripts.benchmark_rag','--all','--e2e','--output',out/'regression.json'))
        command('runtime',pymod('tests.smoke_runtime'))
        command('academic',pymod('tests.smoke_academic_rules'))
        command('security',pymod('tests.smoke_security'))
        # Preserve the old 1,800-case offline diagnostic only when both inputs
        # are available. It is not a semantic pass and never substitutes V3 live.
        if a.bank and a.audit and a.bank.is_file() and a.audit.is_file() and stages['api_1800'] not in ('PASS','COMPLETE_NOT_SEMANTIC_PASS'):
            command('api_1800',[sys.executable,'-B','-X','utf8',str(a.audit),str(a.bank),'--output',str(out/'candidate.csv')])
            stages['api_1800']='COMPLETE_NOT_SEMANTIC_PASS';save()
        elif stages['api_1800']=='NOT_RUN':stages['api_1800']='OPTIONAL_NOT_RUN';save()
        if stages['http']!='PASS' or stages['browser']!='PASS':
            with server(out,False,a.port) as url:
                command('http',pymod('scripts.smoke_live_http','--url',url,'--output',out/'http.json'))
                command('browser',pymod('scripts.smoke_browser','--url',url,'--direct-local','--output',out/'browser.json'))
        if stages['cloud_agent']!='PASS':
            with server(out,True,a.port+1) as url:
                command('cloud_agent',pymod('scripts.smoke_chat_agent_v3','--url',url,'--output',out/'agent_live.json'),cloud=True)
        latest=provenance()
        if any(latest.get(k)!=v for k,v in fingerprint.items()):raise RuntimeError('Source changed during acceptance')
    finally:save()
    required=('python','javascript','regression','runtime','academic','security','http','browser','cloud_agent')
    passed=all(stages[k]=='PASS' for k in required)
    print(json.dumps({'automated_selected_pass':passed,'stages':stages,'source':fingerprint},ensure_ascii=False,indent=2));return int(not passed)

if __name__=='__main__':raise SystemExit(main())
