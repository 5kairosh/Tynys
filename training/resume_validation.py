"""At most 20 complete attempts, 60 seconds between failed downloads."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def verified_count():
    with (ROOT/'training/manifests/validation_manifest.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    count=0
    for row in rows:
        p=ROOT/'training/cache/validation'/(row['sha256']+Path(row['path']).suffix)
        count+=int(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
    return count,len(rows)


def main():
    for attempt in range(1,21):
        before,total=verified_count()
        print(f'Жүктеу әрекеті {attempt}/20. Тексерілгені {before}/{total}.',flush=True)
        result=subprocess.run([sys.executable,str(ROOT/'training/download_validation.py')],cwd=ROOT)
        count,total=verified_count()
        state={'attempt':attempt,'maximum_attempts':20,'verified':count,'required':total,
               'complete':result.returncode==0 and count==total}
        (ROOT/'training/download_progress.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        if state['complete']:
            print('952/952 файл толық тексерілді.',flush=True)
            return
        if attempt<20:
            print('Кэш сақталды. 60 секундтан кейін қайта іске қосылады.',flush=True)
            time.sleep(60)
    report=(f'Жүктеу {count}/{total}-де тоқтады. 20 әрекет аяқталды.\n'
            'T және OOD есептелген жоқ. Жартылай жиын пайдаланылған жоқ.\n'
            'Жалғастыру: site/.venv/Scripts/python training/resume_validation.py\n')
    (ROOT/'training/phase1_download_report.txt').write_text(report,encoding='utf-8')
    raise SystemExit(1)


if __name__=='__main__':
    main()
