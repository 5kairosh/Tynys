"""Build a verified release from Git-tracked project files, excluding local data.

Run AFTER committing final changes. No raw OpenI images, derived error visuals,
OpenI-trained weights, caches, virtual environment or Git database are packaged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path);args=parser.parse_args()
    files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
    forbidden=('site/.venv/','training/cache/','training/artifacts/','training/outputs/multilabel/private/','.git/')
    allowed=[]
    for name in files:
        if not name:continue
        if name in ('presentation/tynys_project_10slides_kz.pptx','presentation/tynys_project_12slides_kz.pptx'):
            continue  # The release contains the current complete 13-slide deck.
        assert not name.startswith(forbidden),name
        assert not any(part in ('.env','__pycache__','.pytest_cache') for part in Path(name).parts),name
        source=(ROOT/name).resolve()
        assert source.is_relative_to(ROOT.resolve()),name
        if source.is_file():allowed.append((name,source))
    assert any(name=='site/model/ood_params.npz' for name,_ in allowed)
    assert any(name=='site/model/image_quality_config.json' for name,_ in allowed)
    assert any(name=='site/static/history-storage.js' for name,_ in allowed)
    assert any(name=='site/model/selective_evaluation.json' for name,_ in allowed)
    assert any(name=='site/static/selective-evaluation.js' for name,_ in allowed)
    assert any(name=='training/ood_stage5/protocol.json' for name,_ in allowed)
    assert any(name=='training/ood_stage5/fit_report.json' for name,_ in allowed)
    for required in ('site/static/generalization-evaluation.js','site/static/research/generalization-stage6.json',
                     'training/generalization_stage6/routing_protocol.json','training/generalization_stage6/routing_evaluation.json',
                     'training/generalization_stage6/DATA_ATTRIBUTION.md','training/GENERALIZATION_STAGE6_KZ.txt'):
        assert any(name==required for name,_ in allowed),required
    for required in ('site/service_runtime.py','training/SERVICE_STAGE7_KZ.txt',
                     'training/verify_service_stage7.py','training/service_stage7/load_local.json'):
        assert any(name==required for name,_ in allowed),required
    for required in ('training/HISTORY_VAULT_STAGE8_KZ.txt','training/history_stage8/verification.json'):
        assert any(name==required for name,_ in allowed),required
    assert any(name=='training/multilabel_colab_kz.ipynb' for name,_ in allowed)
    manifest={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'excluded':'OpenI source images, OpenI-trained weights/error pixels, all training caches, venv, Git and uploaded patient images.',
        'files':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in allowed}}
    args.destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in allowed:z.write(path,'Tynys/'+name)
        z.writestr('Tynys/RELEASE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(args.destination) as z:
        assert z.testzip() is None
        for name,digest in manifest['files'].items():assert hashlib.sha256(z.read('Tynys/'+name)).hexdigest()==digest,name
        assert not any('/private/' in n or '/cache/' in n or '/.venv/' in n for n in z.namelist())
    print(json.dumps({'zip':str(args.destination.resolve()),'files':len(allowed)+1,'bytes':args.destination.stat().st_size,'sha256':hashlib.sha256(args.destination.read_bytes()).hexdigest(),'commit':manifest['commit']},indent=2))


if __name__=='__main__':main()
