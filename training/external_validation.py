"""Reproducible external evaluation. Never tune on external labels.

pip install -r requirements-training.txt
python external_validation.py --manifest external_manifest.csv --internal-dir manifests
Manifest columns: path OR url, label (0/1), patient_id, source.
Only explicit normal vs pneumonia labels are accepted. Other findings are not controls.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'site'))


def download(url, destination):
    import requests
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.part')
    try:
        with requests.get(url, stream=True, timeout=(10, 20)) as r:
            r.raise_for_status()
            with temporary.open('wb') as stream:
                for block in r.iter_content(1024*128):
                    stream.write(block)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def openi_manifest(cache):
    """NLM primary source. MeSH NORMAL or Pneumonia, one frontal view per report.

    Report UID is a patient surrogate, not a verified longitudinal patient ID.
    Explicit patient IDs are preferable for rigorous patient-level evaluation.
    """
    reports = download('https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz', cache/'reports.tgz')
    rows = []
    with tarfile.open(reports) as archive:
        for member in archive:
            if not member.name.endswith('.xml'):
                continue
            doc = ET.fromstring(archive.extractfile(member).read())
            terms = [n.text.strip().lower() for n in doc.findall('.//MeSH/major') if n.text]
            if 'pneumonia' in terms:
                label = 1
            elif terms == ['normal']:
                label = 0
            else:
                continue
            uid = doc.find('uId')
            patient = uid.attrib.get('id') if uid is not None else None
            if not patient:
                continue
            for image in doc.findall('.//parentImage'):
                caption = ' '.join(image.itertext()).lower()
                if not any(word in caption for word in ('frontal', 'pa and', 'pa chest', 'ap chest')):
                    continue
                image_id = image.attrib['id']
                rows.append({'url': f'https://openi.nlm.nih.gov/imgs/512/source/{image_id}.png',
                             'label':label, 'patient_id':'openi:'+patient,
                             'source':'OpenI Indiana, MeSH major label', 'patient_id_kind':'report_uid_surrogate'})
                break
    return rows


def select_groups(rows, count, seed):
    """Sample groups once, no external tuning split and no repeated patients."""
    rng = random.Random(seed)
    groups = {}
    for row in sorted(rows, key=lambda r: (r['patient_id'], r.get('path',r.get('url','')))):
        if not row.get('patient_id') or int(row['label']) not in (0,1):
            raise ValueError('patient_id және 0/1 белгісі міндетті.')
        if row['patient_id'] in groups and int(groups[row['patient_id']]['label']) != int(row['label']):
            raise ValueError('Бір пациенттің белгілері қайшы. Manifest-ті тексер.')
        groups.setdefault(row['patient_id'],row)
    strata = [[r for r in groups.values() if int(r['label']) == label] for label in (0,1)]
    for pool in strata:
        rng.shuffle(pool)
    if min(map(len,strata)) == 0:
        raise ValueError('Сыртқы деректерде екі класс та болуы керек.')
    chosen = strata[0][:count//2]+strata[1][:count//2]
    used = {r['patient_id'] for r in chosen}
    remainder = [r for r in groups.values() if r['patient_id'] not in used]
    rng.shuffle(remainder)
    chosen += remainder[:max(0,count-len(chosen))]
    rng.shuffle(chosen)
    if len(chosen) < 200:
        raise ValueError(f'200 бірегей топқа жетпейді: {len(chosen)}.')
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--openi',action='store_true')
    parser.add_argument('--internal-dir',type=Path,default=ROOT/'training/manifests')
    parser.add_argument('--cache',type=Path,default=ROOT/'training/cache/openi')
    parser.add_argument('--output',type=Path,default=ROOT/'training')
    parser.add_argument('--count',type=int,default=300)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report = args.output/'external_validation_report.txt'
    try:
        if not 200<=args.count<=500:
            raise ValueError('count 200–500 арасында болуы керек.')
        internal_hashes = set()
        for split in ('train','validation','test'):
            with (args.internal_dir/f'{split}_manifest.csv').open(encoding='utf-8-sig') as f:
                internal = list(csv.DictReader(f))
            if not internal or any(not r.get('sha256') for r in internal):
                raise ValueError(f'{split}: ішкі SHA-256 манифесті толық емес.')
            internal_hashes.update(r['sha256'] for r in internal)
        if args.openi:
            rows = openi_manifest(args.cache)
        elif args.manifest:
            with args.manifest.open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
        else:
            raise ValueError('--manifest немесе --openi таңда.')
        rows = select_groups(rows,args.count,args.seed)
        import tensorflow as tf
        import numpy as np
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
        from app import read_config
        from calibration import calibrate_score
        config = read_config(ROOT/'site/model/model_config.json')
        model = tf.keras.models.load_model(ROOT/'site/model/pneumonia_mobilenetv2.keras',compile=False,safe_mode=True)
        seen = set()
        outputs = []
        for i,row in enumerate(rows):
            if row.get('path'):
                path = Path(row['path'])
                if not path.is_absolute():
                    path = args.manifest.parent/path
            else:
                path = download(row['url'],args.cache/(hashlib.sha256(row['url'].encode()).hexdigest()+'.png'))
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if sha in internal_hashes or sha in seen:
                raise ValueError('Сыртқы/ішкі немесе сыртқы суреттер арасында SHA-256 қиылысуы бар.')
            seen.add(sha)
            image = tf.io.decode_image(data,channels=3,expand_animations=False)
            image.set_shape([None,None,3])
            image = tf.image.resize_with_pad(tf.cast(image,tf.float32),*config['image_size'])
            raw = float(model(image[None,...],training=False).numpy()[0,0])
            score = calibrate_score(raw,config.get('calibration'))
            outputs.append({**row,'sha256':sha,'score':score,'prediction':int(score>=config['threshold'])})
            print(f'{i+1}/{len(rows)}',flush=True)
        y = [int(r['label']) for r in outputs]
        pred = [r['prediction'] for r in outputs]
        tn,fp,fn,tp = map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
        result = dict(model='MobileNetV2',cohort='external',count=len(y),threshold=config['threshold'],
                      accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),
                      recall=recall_score(y,pred,zero_division=0),specificity=tn/(tn+fp),
                      f1=f1_score(y,pred,zero_division=0),roc_auc=roc_auc_score(y,[r['score'] for r in outputs]),TN=tn,FP=fp,FN=fn,TP=tp)
        with (ROOT/'site/model/test_metrics.csv').open() as f:
            internal = next(r for r in csv.DictReader(f) if r['model']=='MobileNetV2')
        fields = list(result)
        internal.update(cohort='internal',count=sum(int(float(internal[k])) for k in ('TN','FP','FN','TP')))
        with (args.output/'external_validation_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows([internal,result])
        with (args.output/'external_predictions.csv').open('w',newline='',encoding='utf-8') as f:
            fields=sorted(set().union(*(r.keys() for r in outputs)));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(outputs)
        np.savetxt(args.output/'external_confusion_matrix.csv',[[tn,fp],[fn,tp]],delimiter=',',fmt='%d',header='pred_NORMAL,pred_PNEUMONIA')
        report.write_text('Сыртқы валидация орындалды. Шек өзгертілмеді.\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\nMeSH белгілері клиникалық gold standard емес. Дереккөз және пациент ID түрі external_predictions.csv ішінде. SHA-256 тек байт бойынша көшірмені анықтайды.\n',encoding='utf-8')
    except Exception as exc:
        report.write_text('Сыртқы валидация аяқталмады. Метрикалар есептелген жоқ.\nКедергі: '+str(exc)+'\nБалама: NORMAL/PNEUMONIA, patient_id және path/url бар сыртқы CSV беріп, скриптті қайта орында.\n',encoding='utf-8')
        raise


if __name__=='__main__':
    main()
