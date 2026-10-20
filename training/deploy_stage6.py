"""One-shot, SHA-verified update of this project's existing Codespaces server.

Invoke only in /workspaces/Tynys using an immutable revision and the locally
verified deployment-manifest SHA. No Git, secrets, dependency installs or user
image/history access. On failure restore all replaced files before restarting.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request


def sha(data):return hashlib.sha256(data).hexdigest()


def owned_server(root):
    found=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            command=proc.joinpath('cmdline').read_bytes().split(b'\0')
            if (b'uvicorn' in command and b'app:app' in command and b'--port' in command
                    and command[command.index(b'--port')+1]==b'7860'
                    and proc.joinpath('cwd').resolve()==root/'site'):
                found.append(int(proc.name))
        except (OSError,IndexError):continue
    if len(found)>1:raise ValueError('Multiple project servers; leave them untouched.')
    return found[0] if found else None


def stop_owned(root):
    pid=owned_server(root)
    if pid is None:return
    os.kill(pid,signal.SIGTERM)
    for _ in range(30):
        if not Path('/proc',str(pid)).exists():return
        time.sleep(1)
    raise TimeoutError('Owned server did not stop; no force kill.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--manifest-sha',required=True)
    parser.add_argument('--stage',type=int,choices=(6,7),default=6)
    args=parser.parse_args()
    if sys.platform!='linux' or not re.fullmatch('[0-9a-f]{40}',args.revision) or not re.fullmatch('[0-9a-f]{64}',args.manifest_sha):
        raise ValueError('Run only in the existing Linux Codespace with exact SHA arguments.')
    root=Path('/workspaces/Tynys').resolve()
    if not (root/'.devcontainer/start.py').is_file():raise ValueError('Expected project is absent.')
    prefix='https://raw.githubusercontent.com/5kairosh/Tynys/'+args.revision+'/'
    directory='generalization_stage6' if args.stage==6 else 'service_stage7'
    label='STAGE'+str(args.stage)
    raw=urllib.request.urlopen(prefix+'training/'+directory+'/deployment_manifest.json',timeout=45).read()
    if sha(raw)!=args.manifest_sha:raise ValueError('Deployment manifest SHA mismatch.')
    manifest=json.loads(raw)
    for name,expected in manifest['expected_previous'].items():
        path=(root/name).resolve()
        assert path.is_relative_to(root)
        if path.exists():
            assert sha(path.read_bytes()) in (expected,manifest['files'][name]),'Remote edited since baseline: '+name
        else:
            assert expected is None,'Remote tracked file disappeared: '+name
    def fetch(item):
        name,expected=item;target=(root/name).resolve()
        assert target.is_relative_to(root) and not any(part in ('.env','private','cache','.venv') for part in Path(name).parts)
        data=urllib.request.urlopen(prefix+name,timeout=90).read()
        assert sha(data)==expected,'Download SHA mismatch: '+name
        return name,data
    with ThreadPoolExecutor(max_workers=4) as pool:
        updates=dict(pool.map(fetch,manifest['files'].items()))
    print(label+'_DOWNLOADS_VERIFIED',len(updates),flush=True)
    backup=Path('/tmp/tynys-stage'+str(args.stage)+'-backup')/args.revision
    existing={name:(root/name).exists() for name in updates}
    for name in updates:
        if existing[name]:
            saved=backup/name;saved.parent.mkdir(parents=True,exist_ok=True)
            saved.write_bytes((root/name).read_bytes())
    changed=[];stopped=False
    try:
        stop_owned(root);stopped=True
        for name,data in updates.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.name+'.runtime-update');temporary.write_bytes(data);temporary.replace(path)
            changed.append(name)
        subprocess.run([sys.executable,str(root/'.devcontainer/start.py')],check=True)
        status=json.load(urllib.request.urlopen('http://127.0.0.1:7860/api/status',timeout=20))
        assert status['ready'] and status['comparison_ready'] and status['ensemble']['available'] and status['mcnemar']['available']
        assert status['model_id']==manifest['model_id'] and status['config']['threshold']==.5
        assert status['xray_check']['parameter_id']==manifest['ood_parameter_id']
        assert status['xray_check']['reference_count']==5401 and status['xray_check']['calibration_count']==1352
        audit=status['selective_evaluation'];assert audit['available']
        if args.stage==7:
            assert status['service']['policy']=='bounded_inference_v1'
            assert status['service']['capacity']==3 and status['service']['admitted']==0
            assert json.load(urllib.request.urlopen('http://127.0.0.1:7860/api/ready',timeout=5))['ready']
            assert json.load(urllib.request.urlopen('http://127.0.0.1:7860/api/health',timeout=5))['alive']
        for name,count in [('internal_test',608),('external_openi',167),('rsna_confirmation',348)]:
            assert audit['cohorts'][name]['accepted']['count']==count
        for name,expected in manifest['files'].items():
            assert sha((root/name).read_bytes())==expected,'Published file changed: '+name
        log=Path('/tmp/tynys-codespaces.log').read_text(errors='replace')
        assert 'Traceback' not in log[log.rfind('INFO:     Started server process'):]
        proof={'revision':args.revision,'files':len(updates),'model_id':status['model_id'],
               'ood_parameter_id':status['xray_check']['parameter_id'],'ready':True,
               'audit_available':True,'git_head_advanced':False,'sha_verified':True}
        Path('/tmp/tynys-stage'+str(args.stage)+'-proof.json').write_text(json.dumps(proof))
        print(label+'_DEPLOYED '+json.dumps(proof),flush=True)
    except Exception:
        if stopped:
            stop_owned(root)
            for name in changed:
                path=root/name
                if existing[name]:path.write_bytes((backup/name).read_bytes())
                else:path.unlink(missing_ok=True)
            subprocess.run([sys.executable,str(root/'.devcontainer/start.py')],check=True)
            print(label+'_ROLLED_BACK; production restored. Read the failure above.',flush=True)
        raise


if __name__=='__main__':main()
