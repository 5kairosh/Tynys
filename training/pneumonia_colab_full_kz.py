import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'kagglehub', 'scikit-learn', 'pandas', 'matplotlib', 'Pillow', 'scipy'])


import os, re, json, random, hashlib, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from IPython.display import display
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32                 # Жад жетпесе 16 немесе 8
CNN_EPOCHS = 8
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 5
RUN_BASELINE = True              # CNN мен Transfer Learning-ді салыстыру
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
RESULTS_ROOT = Path('/content/pneumonia_results') if Path('/content').exists() else Path('pneumonia_results')
from datetime import datetime
OUT = RESULTS_ROOT / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True, exist_ok=True)
print('TensorFlow:', tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))
print('Нәтижелер бумасы:', OUT)
if not tf.config.list_physical_devices('GPU'):
    print('GPU жоқ: үйрету баяу жүруі мүмкін. Colab-та GPU таңдауға болады.')


import kagglehub
# Көп жағдайда ашық датасетке кілтсіз кіруге болады.
# Авторизация қатесі болса, жеке келесі ұяшықты орында:
# kagglehub.login()
# Содан кейін осы ұяшықты қайта орында.
download_path = Path(kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia'))

roots = sorted({p.parent for p in download_path.rglob('train')
                if p.is_dir() and (p/'NORMAL').is_dir() and (p/'PNEUMONIA').is_dir()
                and (p.parent/'test'/'NORMAL').is_dir()
                and (p.parent/'test'/'PNEUMONIA').is_dir()}, key=lambda p: len(p.parts))
if not roots:
    raise FileNotFoundError(f'train/NORMAL және test/NORMAL табылмады: {download_path}')
DATA_ROOT = roots[0]
print('Датасет:', DATA_ROOT)


def inferred_group(filename):
    name = Path(filename).stem
    match = re.match(r'(person\d+)', name, re.I)
    if match:
        return 'person:' + match.group(1).lower()
    match = re.match(r'((?:NORMAL2-)?IM-\d+)', name, re.I)
    if match:
        return 'normal:' + match.group(1).lower()
    return 'unknown:' + name

def image_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

records, broken = [], []
for source_split in ['train', 'val', 'test']:
    for label, class_name in enumerate(CLASS_NAMES):
        folder = DATA_ROOT / source_split / class_name
        for path in sorted(folder.glob('*')):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.convert('RGB').load()
                records.append({'path':str(path), 'label':label, 'class':class_name,
                    'source_split':source_split, 'group':inferred_group(path.name),
                    'sha256':image_sha256(path)})
            except (OSError, ValueError) as exc:
                broken.append({'path':str(path), 'error':str(exc)})
raw = pd.DataFrame(records)
if raw.empty:
    raise ValueError('Оқылатын суреттер табылмады.')
print('Оқылатын суреттер саны:', len(raw), '| Бүлінген:', len(broken))
display(pd.crosstab(raw.source_split, raw['class']))
pd.DataFrame(broken, columns=['path','error']).to_csv(OUT/'broken_images.csv', index=False)
if (raw.groupby('sha256').label.nunique() > 1).any():
    raise ValueError('Бірдей суретке әртүрлі белгі берілген. Деректерді қолмен тексер.')

test_df = raw[raw.source_split == 'test'].drop_duplicates('sha256').copy()
dev_df = raw[raw.source_split != 'test'].drop_duplicates('sha256').copy()
test_hashes, test_groups = set(test_df.sha256), set(test_df.group)
bad_groups = set(dev_df.loc[dev_df.sha256.isin(test_hashes), 'group']) | (set(dev_df.group) & test_groups)
removed = dev_df[dev_df.group.isin(bad_groups)]
removed.to_csv(OUT/'excluded_overlap.csv', index=False)
dev_df = dev_df[~dev_df.group.isin(bad_groups)].reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print('Test қиылысуына байланысты development-тен шығарылған суреттер:', len(removed))
if len(dev_df) < 100 or test_df.label.nunique() != 2 or dev_df.label.nunique() != 2:
    raise ValueError('Деректер жеткіліксіз немесе бір класс жоқ. Бумалар мен топтарды тексер.')


# 100 кандидаттан validation көлемі мен класс үлесі ең жақынын таңдау.
# Бұл таңдау тек development белгілерін қолданады; test метрикаларын қолданбайды.
splitter = GroupShuffleSplit(n_splits=100, test_size=0.20, random_state=SEED)
best_split, best_score = None, float('inf')
for train_idx, val_idx in splitter.split(dev_df, dev_df.label, dev_df.group):
    train_part, val_part = dev_df.iloc[train_idx], dev_df.iloc[val_idx]
    if train_part.label.nunique() < 2 or val_part.label.nunique() < 2:
        continue
    score = abs(len(val_part)/len(dev_df)-0.20) + abs(val_part.label.mean()-dev_df.label.mean())
    if score < best_score:
        best_split, best_score = (train_idx, val_idx), score
if best_split is None:
    raise ValueError('Екі класы бар validation бөлу мүмкін болмады.')
train_df = dev_df.iloc[best_split[0]].reset_index(drop=True)
val_df = dev_df.iloc[best_split[1]].reset_index(drop=True)
for left, right in [(train_df,val_df), (train_df,test_df), (val_df,test_df)]:
    assert not (set(left.group) & set(right.group)), 'Топтар қиылысады'
    assert not (set(left.sha256) & set(right.sha256)), 'Сурет көшірмелері қиылысады'
for split, frame in [('train',train_df),('validation',val_df),('test',test_df)]:
    frame.to_csv(OUT/f'{split}_manifest.csv', index=False)
counts = pd.DataFrame({name:frame['class'].value_counts()
    for name,frame in [('train',train_df),('validation',val_df),('test',test_df)]}).fillna(0).astype(int)
display(counts)
counts.T.plot(kind='bar', figsize=(8,4), rot=0, title='Деректердің бөлінуі')
plt.ylabel('Суреттер саны'); plt.tight_layout()
plt.savefig(OUT/'class_distribution.png', dpi=150); plt.show()


fig, axes = plt.subplots(2, 4, figsize=(12,6))
for label in [0,1]:
    examples = train_df[train_df.label==label].sample(4, random_state=SEED)
    for ax, row in zip(axes[label], examples.itertuples()):
        with Image.open(row.path) as im:
            ax.imshow(im.convert('L'), cmap='gray')
        ax.set_title(row._asdict()['class'] if 'class' in row._asdict() else CLASS_NAMES[label])
        ax.axis('off')
plt.tight_layout(); plt.show()


def decode_image(path, label):
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image.set_shape([None,None,3])
    image = tf.image.resize_with_pad(tf.cast(image,tf.float32), *IMG_SIZE)
    return image, tf.cast(label,tf.float32)

def make_dataset(frame, training=False):
    ds = tf.data.Dataset.from_tensor_slices((frame.path.to_numpy(), frame.label.to_numpy()))
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_df, True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)
class_weight = {label:len(train_df)/(2*int((train_df.label==label).sum())) for label in [0,1]}
print('Класс салмақтары:', class_weight)

def augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.02, seed=SEED),
        tf.keras.layers.RandomZoom(0.05, seed=SEED+1),
    ], name='augmentation')

def compile_model(model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy', metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision')])

def callbacks(name):
    return [tf.keras.callbacks.ModelCheckpoint(str(OUT/f'{name}.keras'),
                monitor='val_loss', save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)]

def plot_history(history, name):
    data = pd.DataFrame(history.history)
    data.to_csv(OUT/f'{name}_history.csv', index=False)
    fig, axes = plt.subplots(1,2, figsize=(11,4))
    for ax, metric in zip(axes, ['loss','accuracy']):
        ax.plot(data[metric], label='train')
        ax.plot(data['val_'+metric], label='validation')
        ax.set_title(name+' '+metric); ax.set_xlabel('Epoch (0-ден)'); ax.legend()
    plt.tight_layout(); plt.savefig(OUT/f'{name}_history.png', dpi=150); plt.show()


histories = {}
baseline = None
if RUN_BASELINE:
    inputs = tf.keras.Input(shape=(*IMG_SIZE,3))
    x = augmentation()(inputs)
    x = tf.keras.layers.Rescaling(1./255)(x)
    for filters in [32,64,128]:
        x = tf.keras.layers.Conv2D(filters,3,padding='same',activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    baseline = tf.keras.Model(inputs,outputs,name='baseline_cnn')
    compile_model(baseline,1e-3)
    baseline.summary()
    started = time.time()
    histories['cnn'] = baseline.fit(train_ds, validation_data=val_ds,
        epochs=CNN_EPOCHS, class_weight=class_weight, callbacks=callbacks('cnn_best'))
    print('CNN үйрету уақыты (мин):', round((time.time()-started)/60,2))
    baseline = tf.keras.models.load_model(OUT/'cnn_best.keras')
    plot_history(histories['cnn'],'cnn')


base = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE,3),
    include_top=False, weights='imagenet')
base.trainable = False
inputs = tf.keras.Input(shape=(*IMG_SIZE,3))
x = augmentation()(inputs)
x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
transfer = tf.keras.Model(inputs,outputs,name='mobilenetv2_pneumonia')
compile_model(transfer,1e-3)
transfer.summary()
started = time.time()
histories['head'] = transfer.fit(train_ds, validation_data=val_ds,
    epochs=HEAD_EPOCHS, class_weight=class_weight, callbacks=callbacks('mobilenet_head_best'))
print('Transfer Learning уақыты (мин):', round((time.time()-started)/60,2))
plot_history(histories['head'],'mobilenet_head')
transfer.load_weights(OUT/'mobilenet_head_best.keras')
head_val_loss = float(transfer.evaluate(val_ds,return_dict=True,verbose=0)['loss'])

# 7. Fine-tuning: жоғарыдағы head үйретілгеннен кейін орындалады.
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer,tf.keras.layers.BatchNormalization):
        layer.trainable = False
compile_model(transfer,1e-5)
histories['fine'] = transfer.fit(train_ds, validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS, class_weight=class_weight, callbacks=callbacks('mobilenet_fine_best'))
plot_history(histories['fine'],'mobilenet_fine')
fine_val_loss = float(transfer.evaluate(val_ds,return_dict=True,verbose=0)['loss'])
best_stage = 'fine' if fine_val_loss < head_val_loss else 'head'
chosen_path = OUT/('mobilenet_fine_best.keras' if best_stage=='fine' else 'mobilenet_head_best.keras')
transfer = tf.keras.models.load_model(chosen_path)
print('Таңдалған кезең:',best_stage,'| head val loss:',head_val_loss,'| fine val loss:',fine_val_loss)


def choose_threshold(y_true, probabilities, target_recall=0.90):
    rows = []
    for threshold in np.linspace(0,1,201):
        predicted = (probabilities >= threshold).astype(int)
        tn,fp,fn,tp = confusion_matrix(y_true,predicted,labels=[0,1]).ravel()
        rows.append({'threshold':float(threshold),'recall':tp/(tp+fn),
            'specificity':tn/(tn+fp),'f1':f1_score(y_true,predicted,zero_division=0)})
    table = pd.DataFrame(rows)
    eligible = table[table.recall >= target_recall]
    selected = eligible.sort_values(['specificity','f1','threshold'],ascending=False).iloc[0]
    return float(selected.threshold),table

models = {'MobileNetV2':transfer}
if baseline is not None:
    models = {'CNN':baseline,**models}
thresholds = {}
for name,model in models.items():
    val_prob = model.predict(val_ds,verbose=0).ravel()
    threshold,table = choose_threshold(val_df.label.to_numpy(),val_prob)
    thresholds[name] = threshold
    table.to_csv(OUT/f'{name}_validation_thresholds.csv',index=False)
    print(name,'validation-та таңдалған threshold:',threshold)


# Validation арқылы temperature scaling. Test бұл есептеуге қатыспайды.
from scipy.optimize import minimize_scalar
from scipy.special import expit

def temperature_probabilities(probabilities, temperature):
    probabilities = np.clip(np.asarray(probabilities,dtype=np.float64),1e-7,1-1e-7)
    logits = np.log(probabilities)-np.log1p(-probabilities)
    return expit(logits/temperature)

calibrations = {}
calibration_diagnostics = []
for name,model in models.items():
    raw_val = model.predict(val_ds,verbose=0).ravel()
    val_y = val_df.label.to_numpy()
    logits = np.log(np.clip(raw_val.astype(np.float64),1e-7,1-1e-7))-np.log1p(-np.clip(raw_val.astype(np.float64),1e-7,1-1e-7))
    def nll(temperature):
        z = logits/temperature
        return float(np.mean(np.logaddexp(0,z)-val_y*z))
    fit = minimize_scalar(nll,bounds=(.2,5.),method='bounded')
    if not fit.success:
        raise ValueError('Калибрлеу оңтайландыруы аяқталмады.')
    temperature = float(fit.x)
    calibrated_val = temperature_probabilities(raw_val,temperature)
    threshold,table = choose_threshold(val_y,calibrated_val)
    thresholds[name] = threshold
    calibrations[name] = {'method':'temperature','T':temperature}
    table.to_csv(OUT/f'{name}_validation_calibrated_thresholds.csv',index=False)
    calibration_diagnostics.append({'model':name,'T':temperature,'validation_NLL_raw':nll(1.),'validation_NLL_calibrated':nll(temperature),'threshold':threshold})
pd.DataFrame(calibration_diagnostics).to_csv(OUT/'calibration_validation.csv',index=False)
display(pd.DataFrame(calibration_diagnostics))


# OOD параметрлері тек validation жиынынан есептеледі.
# Бұл эвристиканың non-X-ray деректеріндегі сапасы бөлек тексерілуі керек.
try:
    mobile_layers = transfer.layers
    base_index = next(i for i,layer in enumerate(mobile_layers)
                      if isinstance(layer,tf.keras.Model) and layer.name.startswith('mobilenetv2'))
    features = []
    for images,_ in val_ds:
        x = images
        for layer in mobile_layers[1:base_index+1]:
            x = layer(x,training=False)
        features.append(tf.reduce_mean(x,axis=(1,2)).numpy())
    vectors = np.concatenate(features).astype(np.float64)
    mean = vectors.mean(axis=0)
    covariance = np.cov(vectors,rowvar=False)
    # Shrinkage fixes singular covariance when n < feature dimension.
    scale = max(float(np.trace(covariance)/len(mean)),1e-8)
    covariance = .9*covariance + .1*scale*np.eye(len(mean))
    precision = np.linalg.solve(covariance,np.eye(len(mean)))
    centered = vectors-mean
    distances = np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',centered,precision,centered)))
    ood_threshold = float(np.quantile(distances,.95))
    # Save after the selected model has been serialized, to bind the exact file ID.
    ood_parameters = dict(mean=mean,precision=precision,threshold=ood_threshold)
    print('OOD validation параметрлері есептелді. Non-X-ray валидациясы жүргізілген жоқ.')
except Exception as exc:
    ood_parameters = None
    print('OOD қолжетімсіз:',exc)


test_y = test_df.label.to_numpy()
results, test_probabilities = [], {}
for name,model in models.items():
    raw_prob = model.predict(test_ds,verbose=0).ravel()
    prob = temperature_probabilities(raw_prob,calibrations[name]['T'])
    pred = (prob >= thresholds[name]).astype(int)
    tn,fp,fn,tp = confusion_matrix(test_y,pred,labels=[0,1]).ravel()
    results.append({'model':name,'threshold':thresholds[name],
        'accuracy':accuracy_score(test_y,pred),
        'precision':precision_score(test_y,pred,zero_division=0),
        'recall':recall_score(test_y,pred,zero_division=0),
        'specificity':tn/(tn+fp),'f1':f1_score(test_y,pred,zero_division=0),
        'roc_auc':roc_auc_score(test_y,prob),
        'average_precision':average_precision_score(test_y,prob),
        'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)})
    test_probabilities[name] = prob
    print('\n'+name)
    print(classification_report(test_y,pred,target_names=CLASS_NAMES,zero_division=0))
    fig,axes = plt.subplots(1,3,figsize=(17,4))
    ConfusionMatrixDisplay.from_predictions(test_y,pred,display_labels=CLASS_NAMES,
        cmap='Blues',ax=axes[0],colorbar=False)
    RocCurveDisplay.from_predictions(test_y,prob,ax=axes[1])
    PrecisionRecallDisplay.from_predictions(test_y,prob,ax=axes[2])
    fig.suptitle(name); plt.tight_layout()
    plt.savefig(OUT/f'{name}_test_metrics.png',dpi=150); plt.show()
metrics = pd.DataFrame(results)
metrics.to_csv(OUT/'test_metrics.csv',index=False)
display(metrics.round(4))
predictions = test_df.copy()
for name,prob in test_probabilities.items():
    predictions[name+'_score'] = prob
    predictions[name+'_prediction'] = (prob>=thresholds[name]).astype(int)
predictions.to_csv(OUT/'test_predictions.csv',index=False)


prob = test_probabilities['MobileNetV2']
pred = (prob>=thresholds['MobileNetV2']).astype(int)
errors = np.flatnonzero(pred!=test_y)[:8]
if not len(errors):
    print('Осы test жиынында қате жоқ; бұл клиникалық жарамдылықты дәлелдемейді.')
else:
    fig,axes = plt.subplots(2,4,figsize=(14,7))
    for ax in axes.flat:
        ax.axis('off')
    for ax,index in zip(axes.flat,errors):
        with Image.open(test_df.iloc[index].path) as im:
            ax.imshow(im.convert('L'),cmap='gray')
        ax.set_title(f'Нақты: {CLASS_NAMES[test_y[index]]}\nБолжам: {CLASS_NAMES[pred[index]]}\nБалл: {prob[index]:.3f}')
    plt.tight_layout(); plt.savefig(OUT/'errors.png',dpi=150); plt.show()


MODEL_PATH = OUT/'pneumonia_mobilenetv2.keras'
transfer.save(MODEL_PATH)
if ood_parameters is not None:
    np.savez_compressed(OUT/'ood_params.npz',**ood_parameters,
        model_id=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12])
config = {'classes':CLASS_NAMES,'image_size':list(IMG_SIZE),
    'threshold':thresholds['MobileNetV2'],'calibration':calibrations['MobileNetV2'],
    'model_calibrations':calibrations,'seed':SEED,'selected_stage':best_stage,
    'tensorflow_version':tf.__version__,'dataset':'paultimothymooney/chest-xray-pneumonia',
    'use':'educational prototype; not clinical diagnosis',
    'grouping':'filename-derived surrogate IDs, not verified patient metadata'}
(OUT/'model_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
reloaded = tf.keras.models.load_model(MODEL_PATH)
first_batch = next(iter(test_ds))[0]
np.testing.assert_allclose(transfer.predict(first_batch,verbose=0),
    reloaded.predict(first_batch,verbose=0),rtol=1e-5,atol=1e-6)
print('Модель сақталды және қайта жүктеу тексерілді:',MODEL_PATH)

def predict_xray(filepath, model=reloaded, threshold=config['threshold']):
    image,_ = decode_image(tf.constant(str(filepath)),tf.constant(0))
    raw_score = float(model.predict(image[None,...],verbose=0)[0,0])
    score = float(temperature_probabilities([raw_score],config['calibration']['T'])[0])
    label = CLASS_NAMES[int(score>=threshold)]
    with Image.open(filepath) as original:
        plt.figure(figsize=(5,5)); plt.imshow(original.convert('L'),cmap='gray')
    plt.title(f'Модель санаты: {label}\nПневмония баллы: {score:.3f}; шек: {threshold:.3f}')
    plt.axis('off'); plt.show()
    print('Бұл оқу моделінің бағасы; медициналық диагноз емес.')
    return {'model_class':label,'pneumonia_score':score,'threshold':threshold}

# Алдымен датасеттегі бір суреттің өңделуін тексеру.
predict_xray(test_df.iloc[0].path)


# Өз рентген суретіңді тексеру үшін True қойып, осы ұяшықты қайта орында.
UPLOAD_OWN_XRAY = False
if UPLOAD_OWN_XRAY:
    # Бұл ұяшықты тек өз суретіңді сынағың келгенде орында.
    from google.colab import files
    uploaded = files.upload()
    for filename in uploaded:
        if Path(filename).suffix.lower() in {'.jpg','.jpeg','.png'}:
            try:
                print(predict_xray(filename))
            except (OSError,tf.errors.InvalidArgumentError) as exc:
                print('Суретті оқу мүмкін болмады:',filename,exc)
        else:
            print('Қолдау көрсетілетін форматтар: JPG, JPEG, PNG.')
else:
    print('Жеке сурет жүктеу өткізіліп жіберілді. Қосу үшін UPLOAD_OWN_XRAY = True қой.')


row = metrics[metrics.model=='MobileNetV2'].iloc[0]
summary = f"""Өкпе рентгенінен пневмония белгілерін анықтау — оқу жобасы

Датасет: Kaggle Chest X-Ray Images (Pneumonia).
Оқылған жарамды суреттер: {len(raw)}.
Train: {len(train_df)}, validation: {len(val_df)}, test: {len(test_df)}.
Development-тен test қиылысуына байланысты шығарылған суреттер: {len(removed)}.
Модель: MobileNetV2, Transfer Learning; таңдалған кезең: {best_stage}.
Шешім шегі validation жиынында таңдалды: {row['threshold']:.3f}.

Test accuracy: {row['accuracy']:.2%}
Test precision: {row['precision']:.2%}
Test recall/sensitivity: {row['recall']:.2%}
Test specificity: {row['specificity']:.2%}
Test F1: {row['f1']:.4f}
Test ROC-AUC: {row['roc_auc']:.4f}
Test average precision: {row['average_precision']:.4f}
False negative: {int(row['FN'])}; false positive: {int(row['FP'])}.

Тұжырым: бұл нәтижелер тек осы эксперименттің бөлінуіне қатысты.
Файл атынан алынған топтар ресми пациент ID емес.
Клиникалық қолдануға сыртқы деректе және мамандармен бөлек тексеру қажет.
90% accuracy мақсатынан өтті ме: {'иә' if row['accuracy']>=0.90 else 'жоқ'}.
"""
print(summary)
(OUT/'project_summary.txt').write_text(summary,encoding='utf-8')


"""Exact paired McNemar test, requiring row-aligned predictions."""
import csv
import hashlib
import json
from pathlib import Path
from scipy.stats import binomtest


def calculate_mcnemar(predictions, model_dir):
    with Path(predictions).open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('Болжамдар бос немесе суреттер қайталанған.')
    b = c = 0
    for row in rows:
        truth = int(row['label'])
        cnn, mobile = int(row['CNN_prediction']), int(row['MobileNetV2_prediction'])
        if any(x not in (0,1) for x in (truth,cnn,mobile)):
            raise ValueError('Белгі 0/1 болуы керек.')
        b += int(cnn == truth and mobile != truth)
        c += int(cnn != truth and mobile == truth)
    p = float(binomtest(b,b+c,.5,alternative='two-sided').pvalue) if b+c else 1.
    result = {'available':True,'method':'exact McNemar (two-sided binomial)',
              'n':len(rows),'cnn_only_correct':b,'mobilenet_only_correct':c,'p_value':p,
              'alpha':.05,'significant':p<.05,
              'notice':'Жұптық тест, осы ішкі test жиынына қатысты. Пациент суреттері тәуелсіз болмауы мүмкін.'}
    result['model_ids'] = {name:hashlib.sha256((Path(model_dir)/file).read_bytes()).hexdigest()[:12]
                           for name,file in [('CNN','cnn_best.keras'),('MobileNetV2','pneumonia_mobilenetv2.keras')]}
    result['config_sha256'] = hashlib.sha256((Path(model_dir)/'model_config.json').read_bytes()).hexdigest()
    return result



paired_test = calculate_mcnemar(OUT/'test_predictions.csv',OUT)
(OUT/'mcnemar.json').write_text(json.dumps(paired_test,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'project_summary.txt').open('a',encoding='utf-8') as stream:
    stream.write('\nMcNemar (жұптық тест): '+json.dumps(paired_test,ensure_ascii=False)+'\n')
print(paired_test)


archive = shutil.make_archive(str(OUT),'zip',root_dir=OUT)
print('Нәтижелер архиві:',archive)
from google.colab import files
files.download(archive)
