"""Жеке оқу эксперименті. Сайттың binary модельдерін өзгертпейді.

OpenI деректері, салмақтар және қате суреттері тек жергілікті кэш/private
бумасында қалады. Оларды Git/ZIP-ке қоспау керек. Белгілер — MeSH report
деңгейіндегі әлсіз белгілер, расталған әр сурет диагнозы емес.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
import threading
_LOCAL=threading.local()

LABELS=('pneumonia','atelectasis','cardiomegaly')
SEED=42
ROOT=Path('/content/tynys_multilabel')


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def fetch(url,path):
    import requests
    if path.exists():return path
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.parent.name=='images':
        shared=path.parents[2]/'openi'/path.name
        if shared.exists():
            import shutil
            shutil.copyfile(shared,path);return path
    if not hasattr(_LOCAL,'session'):_LOCAL.session=requests.Session()
    attempts=2 if path.parent.name=="images" else 4
    for attempt in range(attempts):
        try:
            response=_LOCAL.session.get(url,timeout=(10,25));response.raise_for_status()
            temporary=path.with_suffix('.part');temporary.write_bytes(response.content);temporary.replace(path)
            return path
        except requests.RequestException:
            if attempt==attempts-1:raise
            time.sleep(2*(attempt+1))


def terms(record):
    mesh=record.get('MeSH') or {}
    aliases={'pulmonary atelectasis':'atelectasis'}
    raw={re.sub('<[^>]*>','',t).strip().lower().split('/')[0].strip() for t in (mesh.get('major') or [])+(mesh.get('minor') or [])}
    return {aliases.get(t,t) for t in raw}


def prepare(root,per_label=100):
    cache=root/'training/cache/openi_multilabel';out=root/'training/outputs/multilabel'
    out.mkdir(parents=True,exist_ok=True)
    groups={}
    for query in (*LABELS,'normal'):
        path=fetch('https://openi.nlm.nih.gov/api/search?query='+query+'&m=1&n=1000&coll=cxr',cache/(query+'.json'))
        payload=json.loads(path.read_text(encoding='utf-8'))
        for r in payload.get('list',[]):
            if not r.get('uid') or not r.get('imgLarge'):continue
            item=groups.setdefault(r['uid'],{'terms':set(),'urls':set()})
            item['terms'].update(terms(r));item['urls'].add('https://openi.nlm.nih.gov'+r['imgLarge'])
        print(f'Metadata {query}: {len(payload.get("list",[]))} records',flush=True)
    rng=random.Random(SEED);chosen=set()
    # Deterministic cohort quotas use report labels before any model is trained.
    for label in (*LABELS,'normal'):
        pool=sorted(uid for uid,g in groups.items() if label in g['terms'])
        rng.shuffle(pool);chosen.update(pool[:per_label])
        print(f'Cohort {label}: {min(len(pool),per_label)} reports',flush=True)
    pending=[]
    for uid in sorted(chosen):
        g=groups[uid]
        for url in sorted(g['urls'])[:2]:
            relative='training/cache/openi_multilabel/images/'+hashlib.sha256(url.encode()).hexdigest()+'.png'
            pending.append({'report_id':uid,'url':url,'cache_path':relative,**{label:int(label in g['terms']) for label in LABELS}})
    def load(row):
        import requests
        try:
            data=fetch(row['url'],root/row['cache_path']).read_bytes()
            from PIL import Image
            with Image.open(root/row['cache_path']) as im:im.verify()
            return {**row,'sha256':hashlib.sha256(data).hexdigest()}
        except (requests.RequestException, OSError) as exc:
            return {**row,'excluded_reason':type(exc).__name__}
    rows=[];excluded=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,row in enumerate(pool.map(load,pending),1):
            if 'excluded_reason' in row:excluded.append(row)
            else:rows.append(row)
            if i%50==0 or i==len(pending):print(f'Multi-label images {i}/{len(pending)}',flush=True)
    # Availability filtering happens before splitting or fitting, and is reported.
    assert len(rows)>=200,'Too few available source images for this prototype.'
    if excluded:write_csv(out/'unavailable_sources.csv',excluded)
    downloaded=len(rows)
    # Union report IDs linked by a byte-identical image BEFORE splitting.
    parent={r['report_id']:r['report_id'] for r in rows}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    hashes={}
    for row in rows:
        previous=hashes.setdefault(row['sha256'],row['report_id'])
        a,b=find(previous),find(row['report_id'])
        if a!=b:parent[max(a,b)]=min(a,b)
    unique={}
    for row in rows:
        row['group_id']=find(row['report_id'])
        if row['sha256'] in unique:
            for label in LABELS:unique[row['sha256']][label]=max(unique[row['sha256']][label],row[label])
        else:unique[row['sha256']]=row
    rows=list(unique.values())
    # Stratify report components by complete multi-hot signature. Rare strata
    # go to train; no search over test metrics or re-splitting after evaluation.
    buckets={}
    for group in sorted({r['group_id'] for r in rows}):
        members=[r for r in rows if r['group_id']==group]
        signature=tuple(max(r[label] for r in members) for label in LABELS)
        buckets.setdefault(signature,[]).append(group)
    assignments={}
    for signature,pool in sorted(buckets.items()):
        rng.shuffle(pool);n=len(pool)
        nv=max(1,round(n*.15)) if n>=7 else 0
        nt=max(1,round(n*.15)) if n>=7 else 0
        for index,group in enumerate(pool):assignments[group]='validation' if index<nv else 'test' if index<nv+nt else 'train'
    splits={name:[{**r,'split':name} for r in rows if assignments[r['group_id']]==name] for name in ('train','validation','test')}
    audit_splits(splits)
    for name,part in splits.items():write_csv(out/(name+'_manifest.csv'),part)
    overlap=0
    binary=root/'training/external_predictions.csv'
    if binary.exists():
        with binary.open(encoding='utf-8-sig') as f:overlap=len({r['sha256'] for r in rows}&{r['sha256'] for r in csv.DictReader(f)})
    report={'seed':SEED,'labels':list(LABELS),'source':'NLM OpenI Indiana API; MeSH major+minor report labels',
        'splits':{name:{'images':len(part),'groups':len({r['group_id'] for r in part}),'positive_counts':{l:sum(r[l] for r in part) for l in LABELS}} for name,part in splits.items()},
        'requested_images':len(pending),'unavailable_images':len(excluded),'downloaded_images':downloaded,'images_with_multiple_positive_labels':sum(sum(int(r[l]) for l in LABELS)>=2 for r in rows),'license_url':'https://creativecommons.org/licenses/by-nc-nd/4.0/','label_aliases':{'Pulmonary Atelectasis':'atelectasis'},'unique_images':len(rows),'duplicates_removed':downloaded-len(rows),
        'group_overlap':0,'sha256_overlap':0,'binary_external_cohort_image_overlap':overlap,
        'limitations':['Small quota-sampled prototype cohort; not population representative.','Report UID is not a verified longitudinal patient ID.','Absent MeSH label treated as weak negative; labels not image-level clinical ground truth.','Frontal/lateral projections mixed; no PA-only claim.','Separate task; may overlap prior binary external cohort, not new independent binary evidence.'],
        'distribution':'OpenI images, image-derived error visualizations and trained weights remain local only; excluded from Git and ZIP per project requirements.'}
    (out/'data_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return splits


def audit_splits(splits):
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for key in ('group_id','report_id','sha256'):
            assert not {r[key] for r in splits[a]} & {r[key] for r in splits[b]},(a,b,key)
    for name,part in splits.items():
        assert len(part)==len({r['sha256'] for r in part})
        for label in LABELS:assert {int(r[label]) for r in part}=={0,1},(name,label,'both classes required')


def choose_threshold(y,p):
    import numpy as np
    from sklearn.metrics import f1_score
    # Per-label F1 only on validation; ties prefer the larger threshold.
    return max((f1_score(y,p>=t,zero_division=0),float(t)) for t in np.linspace(.01,.99,99))[1]


def train(root,splits,head_epochs=3,fine_epochs=2):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import tensorflow as tf
    import numpy as np
    from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score,precision_score
    tf.keras.utils.set_random_seed(SEED)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    out=root/'training/outputs/multilabel';private=out/'private';private.mkdir(exist_ok=True)
    def dataset(part,training=False):
        paths=[str(root/r['cache_path']) for r in part]
        y=np.array([[int(r[l]) for l in LABELS] for r in part],dtype=np.float32)
        ds=tf.data.Dataset.from_tensor_slices((paths,y))
        if training:ds=ds.shuffle(len(part),seed=SEED,reshuffle_each_iteration=True)
        def decode(path,label):
            im=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False);im.set_shape([None,None,3])
            return tf.image.resize_with_pad(tf.cast(im,tf.float32),224,224),label
        return ds.map(decode,num_parallel_calls=2).batch(16).prefetch(1)
    train_ds=dataset(splits['train'],True);val_ds=dataset(splits['validation']);test_ds=dataset(splits['test'])
    inputs=tf.keras.Input((224,224,3))
    x=tf.keras.layers.Rescaling(1/127.5,offset=-1)(inputs)
    base=tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
    base.trainable=False
    x=base(x,training=False);x=tf.keras.layers.GlobalAveragePooling2D()(x);x=tf.keras.layers.Dropout(.2)(x)
    heads=[tf.keras.layers.Dense(1,activation='sigmoid',name=label)(x) for label in LABELS]
    model=tf.keras.Model(inputs,tf.keras.layers.Concatenate(name='multi_label_scores')(heads))
    def fit_stage(stage,epochs,lr):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss=tf.keras.losses.BinaryCrossentropy(),metrics=[tf.keras.metrics.AUC(multi_label=True,num_labels=len(LABELS),name='auc')])
        history=model.fit(train_ds,validation_data=val_ds,epochs=epochs,verbose=2,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)])
        return {'stage':stage,**history.history}
    history=[fit_stage('head',head_epochs,1e-3)]
    best_loss=min(history[0]['val_loss']);model.save_weights(private/'head.weights.h5')
    base.trainable=True
    for layer in base.layers[:-30]:layer.trainable=False
    for layer in base.layers:
        if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
    history.append(fit_stage('fine_last30_bn_frozen',fine_epochs,1e-5))
    if min(history[1]['val_loss'])>best_loss:
        model.load_weights(private/'head.weights.h5');selected='head'
    else:selected='fine'
    validation=model.predict(val_ds,verbose=0)
    val_y=np.array([[int(r[l]) for l in LABELS] for r in splits['validation']])
    thresholds={l:choose_threshold(val_y[:,i],validation[:,i]) for i,l in enumerate(LABELS)}
    # Freeze model and thresholds before opening test labels for evaluation.
    model.save(private/'openi_multilabel.keras')
    probabilities=model.predict(test_ds,verbose=0)
    test_y=np.array([[int(r[l]) for l in LABELS] for r in splits['test']])
    metrics=[]
    for i,label in enumerate(LABELS):
        pred=probabilities[:,i]>=thresholds[label]
        tn,fp,fn,tp=map(int,confusion_matrix(test_y[:,i],pred,labels=[0,1]).ravel())
        metrics.append({'label':label,'threshold':thresholds[label],'roc_auc':roc_auc_score(test_y[:,i],probabilities[:,i]),'f1':f1_score(test_y[:,i],pred,zero_division=0),'precision':precision_score(test_y[:,i],pred,zero_division=0),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'TN':tn,'FP':fp,'FN':fn,'TP':tp,'count':len(pred)})
    write_csv(out/'per_label_metrics.csv',metrics)
    predictions=[]
    for r,score in zip(splits['test'],probabilities):predictions.append({**{k:r[k] for k in ('report_id','group_id','sha256')},**{l:int(r[l]) for l in LABELS},**{l+'_score':float(score[i]) for i,l in enumerate(LABELS)}})
    write_csv(out/'test_predictions.csv',predictions)
    errors=[]
    for j,r in enumerate(splits['test']):
        for i,label in enumerate(LABELS):
            if int(probabilities[j,i]>=thresholds[label])!=test_y[j,i]:
                errors.append((abs(float(probabilities[j,i])-thresholds[label]),j,i))
    errors.sort(reverse=True);chosen=[];seen=set()
    for _,j,i in errors:
        if j not in seen:chosen.append((j,i));seen.add(j)
        if len(chosen)==8:break
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    for number,(j,i) in enumerate(chosen,1):
        r=splits['test'][j];fig,ax=plt.subplots(figsize=(6,7))
        with Image.open(root/r['cache_path']) as im:ax.imshow(im.convert('L'),cmap='gray')
        ax.axis('off');label=LABELS[i]
        ax.set_title(f'{label}: MeSH={test_y[j,i]}, predicted={int(probabilities[j,i]>=thresholds[label])}\nscore={probabilities[j,i]:.3f}; threshold={thresholds[label]:.2f}')
        fig.text(.04,.025,'Report-level weak label; educational error review only.',fontsize=9)
        fig.savefig(private/f'error_{number:02}.png',dpi=130,bbox_inches='tight');plt.close(fig)
    report={'completed':True,'seed':SEED,'labels':list(LABELS),'selected_stage':selected,'threshold_selection':'per-label validation F1, 0.01..0.99 grid; no test tuning','thresholds':thresholds,'history':history,'metrics':metrics,'error_visualizations':len(chosen),'weights_sha256':hashlib.sha256((private/'openi_multilabel.keras').read_bytes()).hexdigest(),'trained_on':'OpenI Indiana, weak report MeSH labels','distributed_weights':False,'website_integration':False,'architecture':'ImageNet MobileNetV2 + GAP + Dropout(.2) + 3 independent Dense(1,sigmoid) heads, concatenated; BCE; final30 layers fine-tuned with BN frozen','tensorflow_version':tf.__version__}
    (out/'experiment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--per-label',type=int,default=100);parser.add_argument('--head-epochs',type=int,default=3);parser.add_argument('--fine-epochs',type=int,default=2);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--reuse-manifests',action='store_true')
    args=parser.parse_args()
    if args.reuse_manifests:
        splits={}
        for name in ('train','validation','test'):
            with (args.root/f'training/outputs/multilabel/{name}_manifest.csv').open(encoding='utf-8-sig') as f:splits[name]=list(csv.DictReader(f))
        audit_splits(splits)
        for part in splits.values():
            for r in part:assert hashlib.sha256((args.root/r['cache_path']).read_bytes()).hexdigest()==r['sha256']
    else:splits=prepare(args.root,args.per_label)
    if not args.prepare_only:train(args.root,splits,args.head_epochs,args.fine_epochs)


if __name__=='__main__':main()
