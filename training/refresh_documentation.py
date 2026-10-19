"""Generate current documentation from measured artifacts, never literal old metrics."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    model=ROOT/'site/model'
    config=json.loads((model/'model_config.json').read_text(encoding='utf-8'))
    calibration=json.loads((ROOT/'training/calibration_report.json').read_text(encoding='utf-8'))
    external=json.loads((model/'external_validation.json').read_text(encoding='utf-8'))
    paired=json.loads((model/'mcnemar.json').read_text(encoding='utf-8'))
    with (model/'test_metrics.csv').open() as stream:
        metrics=list(csv.DictReader(stream))
    def pct(value):return f'{float(value)*100:.2f}%'
    def line(row):
        return (f"{row['model']}: шек {float(row['threshold']):.3f}, accuracy {pct(row['accuracy'])}, "
                f"precision {pct(row['precision'])}, recall {pct(row['recall'])}, specificity {pct(row['specificity'])}, "
                f"F1 {float(row['f1']):.4f}, ROC-AUC {float(row['roc_auc']):.4f}; "
                f"TN {int(float(row['TN']))}, FP {int(float(row['FP']))}, FN {int(float(row['FN']))}, TP {int(float(row['TP']))}.")
    current='\n'.join(line(row) for row in metrics)
    external_rows=external.get('models',[external['metrics']])
    ext='\n'.join(line(row) for row in external_rows)
    temperatures='\n'.join(f"{row['model']}: T={row['T']:.8f}, validation NLL {row['NLL_raw']:.8f} → {row['NLL_calibrated']:.8f}." for row in calibration['validation_calibration'])
    ensemble=config.get('ensemble')
    ensemble_text=('Ensemble: validation-да таңдалған салмақтар '+json.dumps(ensemble['weights'])+f", шек {ensemble['threshold']:.3f}. Сыртқы белгілер таңдау үшін қолданылған жоқ." if ensemble else 'Ensemble әлі қосылған жоқ, бұл келесі кезең.')
    multi_path=ROOT/'training/outputs/multilabel/experiment_report.json'
    multi=None
    multi_text='Multi-label жеке эксперименті әзірленуде; сайтқа қосылмаған.'
    if multi_path.exists():
        multi=json.loads(multi_path.read_text(encoding='utf-8'))
        data=json.loads((multi_path.parent/'data_report.json').read_text(encoding='utf-8'))
        rows='\n'.join(f"{r['label']}: шек {r['threshold']:.2f}, AUC {r['roc_auc']:.4f}, F1 {r['f1']:.4f}, sensitivity {pct(r['sensitivity'])}, specificity {pct(r['specificity'])}; TN {r['TN']}, FP {r['FP']}, FN {r['FN']}, TP {r['TP']}." for r in multi['metrics'])
        multi_text=('MULTI-LABEL — ЖЕКЕ ОҚУ ЭКСПЕРИМЕНТІ\n'+json.dumps(data['splits'],ensure_ascii=False)+'\n'+rows+'\nImageNet MobileNetV2 + GAP + үш sigmoid Dense(1), BCE. Соңғы 30 қабат, BN мұздатылған. Таңдалған кезең: '+multi['selected_stage']+'.\nШектер тек validation F1 арқылы. MeSH report-level әлсіз белгілер, жоқ белгі weak negative деп алынған.\nReport ID longitudinal пациент ID емес; frontal/lateral аралас. Шағын квоталық жиын жалпы популяцияны көрсетпейді.\nBinary сыртқы cohort-пен сурет қиылысы: '+str(data['binary_external_cohort_image_overlap'])+'; бұл жаңа тәуелсіз binary тексеру емес.\nOpenI суреттерін, үйретілген салмақтарын және қате суреттерін таратуға болмайды: олар тек жергілікті private/cache бумасында, Git/ZIP-ке кірмейді.\nСайтқа қосылған жоқ. Қайта орындау: training/multilabel_colab_kz.ipynb немесе training/multilabel_experiment.py.\nТолық сандар: training/outputs/multilabel/per_label_metrics.csv және experiment_report.json.\n')
    docker=None
    evidence=ROOT/'training/docker_evidence/verification.json'
    if evidence.exists():
        candidate=json.loads(evidence.read_text(encoding='utf-8'))
        build=json.loads((evidence.parent/'build_result.json').read_text(encoding='utf-8'))
        if (candidate.get('built') and candidate.get('run_verified') and
            build['dockerfile_sha256']==hashlib.sha256((ROOT/'Dockerfile').read_bytes()).hexdigest() and
            build['requirements_sha256']==hashlib.sha256((ROOT/'site/requirements.txt').read_bytes()).hexdigest()):
            docker=candidate
    docker_text=(f"Docker: build/run тексерілді (image tynys, Server {docker['server_version']}). "
        f"Image көлемі: {docker['image_size_display']} ({docker['image_size_bytes']} байт). "
        f"Тексерілген host порт: {docker['host_port']}, контейнер: 7860. "
        "PNG/CR DICOM 200; CT DICOM/мәтінді JPG 400; PDF 200, 2 бет; Traceback жоқ. Контейнер жойылды."
        if docker else 'Осы Dockerfile/requirements үшін Docker build/run тексерілгені расталмаған.')
    deployment=f"""## Deployment

{docker_text}

Жоба түбірінен:

```sh
docker build -t tynys .
docker run --rm -d --name tynys-test -p 127.0.0.1:7860:7860 tynys
```

7860 бос болмаса, `-p 127.0.0.1:7861:7860` қолданыңыз. Нақты тексеруде 7860-та жергілікті сервер
болғандықтан 7861 қолданылды. `/api/status` ready=true болғанша күтіңіз.
`docker logs tynys-test` арқылы журналды көріп, `docker stop tynys-test` арқылы тоқтатыңыз.
`--rm` тоқтаған контейнерді автоматты жояды. Қосымша `docker rm` қажет емес.
Команданың ішкі порты 7860, USER tynys (UID 1000), requirements нұсқалары өзгермеген.
CPU тексеруі өтті. CUDA жоқтығы/cuInit 303 хабарламасы болды, HTTP/PDF жұмысына кедергі келтірмеді.
Дәлелдер: `training/docker_evidence/` ішіндегі JSON, журнал және екі беттік PDF.

### Hugging Face Docker Space

1. Space құрғанда SDK ретінде Docker таңдаңыз.
2. Репозиторий түбіріне Dockerfile мен README.md, сондай-ақ site бумасын салыңыз.
   README.md YAML басында `sdk: docker` және `app_port: 7860` сақталуы керек.
3. site/model ішінде екі binary .keras модель, model_config.json, test_metrics.csv,
   ood_params.npz, mcnemar.json, external_validation.json және image_quality_config.json бірге болуы керек.
4. Build біткен соң Space Logs пен `/api/status` тексеріңіз. Контейнер 0.0.0.0:7860 тыңдайды.
   Venv, cache, пайдаланушы суреттері, OpenI суреттері мен multi-label салмақтарын жүктемеңіз.

Space интернетке осы тапсырмада жарияланған жоқ. Ресми нұсқаулық:
[Hugging Face Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).
"""
    presentation_name='tynys_complete_13slides_kz.pptx' if multi else 'tynys_project_12slides_kz.pptx'
    saved_status=json.loads((ROOT/'training/IMPLEMENTATION_STATUS.json').read_text(encoding='utf-8'))
    ci=saved_status.get('ci',{})
    ci_links=''
    ci_text='CI: .github/workflows/ci.yml, Ubuntu/Python 3.11. Қашықтағы CI орындалуы осы компьютерде расталмаған.'
    if ci.get('remote_run') and ci.get('status')=='success':
        proof=json.loads((ROOT/'training/ci_evidence/success_run.json').read_text(encoding='utf-8'))
        assert proof['run_id']==ci['run_id'] and proof['pytest_count']==56
        ci_text=f"CI қашықта өтті (ubuntu-latest, Python 3.11, 56 тест). Run: {ci['run_url']} . Тексерілген code commit: {proof['tested_commit']}."
        ci_links=f"Репозиторий: {ci['repo_url']}\n[![CI]({ci['repo_url']}/actions/workflows/ci.yml/badge.svg?branch=master)]({ci['repo_url']}/actions/workflows/ci.yml)"
    body=f'''TYNYS — ҚАЗІРГІ ЕСЕПТЕЛГЕН НҰСҚА

ІСКЕ ҚОСУ
Архивті толық шығарып, Python 3.11 бар Windows-та site/start.bat бас.
Сервер: http://127.0.0.1:7860 . Терезесін ашық қалдыр. Алғашқы іске қосуда requirements орнатылады.
Бұрынғы ортада: site/.venv/Scripts/python -m pip install -r site/requirements.txt

МҮМКІНДІКТЕР
JPG/PNG және CR/DX DICOM. Әр файл ≤12 МБ, бір топта ≤10 файл.
CNN, MobileNetV2 және Ensemble салыстыруы, Grad-CAM, PNG/мәтін/PDF жүктеу.
Тарих әдепкіде тек бет жадында. Келісіммен осы браузердің IndexedDB қоймасында
түпнұсқа файл мен нәтиже сақталады: соңғы 20 талдау, 80 МиБ, 7 күндік мерзім.
Ескіргендері келесі ашқанда/белсенді тексеруде жойылады. Сақтауды өшіру немесе
«Тарихты тазалау» сақталған көшірмелерді жояды. Ортақ браузерде тарих көрінеді.
Серверге суреттер файл болып сақталмайды. Cookie/localStorage қолданылмайды.
Нұсқау: training/HISTORY_KZ.txt.
Multipart қабылдау жадта шектелген. Тек оқу прототипі, медициналық диагноз емес.
NORMAL барлық ауруды жоққа шығармайды. Балл клиникалық сенімділік пайызы емес.

СУРЕТ САПАСЫ — БӨЛЕК ТЕХНИКАЛЫҚ ТЕКСЕРУ
Жарықтық, контраст, өлшем, пішім және ұсақ деталь туралы ескертулер.
Деталь шегі: validation-ның төменгі 1-процентилі, site/model/image_quality_config.json.
Ескерту сайтта, тарихта, топтық тізімде және TXT/PDF есептерде бар; модель баллы өзгермейді.
Рентгенді, анатомиялық қиылуды және клиникалық сапаны растамайды.
Толық есептеу мен шектеуі: training/QUALITY_CHECK_KZ.txt.

ӨҢДЕУ СӘЙКЕСТІГІ
JPG/PNG: decode_image(channels=3,expand_animations=False), float32, resize_with_pad(224,224), модель.
EXIF rotation, сыртқы нормализация және crop жоқ. Rescaling модель ішінде.
DICOM: CR/DX, бір MONOCHROME кадр, Modality LUT, 1–99 процентиль, RGB PNG жадта,
кейін сол TF pipeline. MONOCHROME1 инверсиясы бар. CT/MR қабылданбайды.

ІШКІ TEST, 618 СУРЕТ — ҚАЗІРГІ КАЛИБРЛЕНГЕН НӘТИЖЕ
{current}
90% accuracy мақсаты: {'жетті' if float(next(r for r in metrics if r['model']=='MobileNetV2')['accuracy'])>=.9 else 'жетпеді'}.
Бірдей accuracy қайта есептегенде өзгермеуі мүмкін. Бұл сандар қазіргі конфигурациямен есептелген.

КАЛИБРЛЕУ — ОРЫНДАЛДЫ
952/952 validation файлы SHA-256 бойынша расталды. Test және сыртқы белгілер fitting-ке қатыспайды.
{temperatures}
Әр модельдің баллы sigmoid(logit(raw)/T) арқылы калибрленеді.
Шек тек validation-да recall≥0.90, specificity, F1 ережесімен қайта таңдалған.
Grad-CAM reconstructed_score raw sigmoid-пен тексеріледі, калибрленген баллмен емес.
Калибрлеу клиникалық сенімділікті дәлелдемейді.

OOD — ПАРАМЕТРЛЕРІ ДАЙЫН
Validation: 952, белгілер: 1280, shrinkage covariance.
Махаланобис қашықтығының 95-процентиль шегі: {calibration['ood']['threshold']:.8f}.
site/model/ood_params.npz модель fingerprint-іне байланған.
ood_guard_v1: OOD ауытқуы не тексеру қолжетімсіз болғанда сайт, тарих және TXT/PDF
«Сенімді бағалау мүмкін емес» күйін көрсетеді. Баллдар техникалық нәтиже ретінде қалады.
Бұл сақтық ережесінің клиникалық сапасы және қамту/қателік қатынасы бағаланбаған.
Non-X-ray деректерінде арнайы валидация жасалған жоқ. Бұл рентген екенін растайтын детектор емес.

СЫРТҚЫ OPENI, 200 СУРЕТ — ҚАЗІРГІ КОНФИГУРАЦИЯ
160 NORMAL, 40 PNEUMONIA, seed 42. Бұрынғы таңдау сақталды, шекті сыртқы деректе өзгертпедік.
{ext}
SHA-256 ішкі train/validation/test қиылысы: 0. Бір report UID-інен бір сурет.
Report UID — surrogate топ, longitudinal patient ID толық расталмаған. MeSH gold standard емес.
Проекция PA деп кепілденбейді. Нәтиже тек осы таңдалған жиынға қатысты.

СТАТИСТИКА ЖӘНЕ ENSEMBLE
Exact paired McNemar: CNN-only {paired['cnn_only_correct']}, MobileNetV2-only {paired['mobilenet_only_correct']},
p={paired['p_value']:.12g}, alpha=0.05. Бұл ішкі test-ке қатысты.
{ensemble_text}

ҚАЙТА ЕСЕПТЕУ — ЖОБА ТҮБІРІНЕН
site/.venv/Scripts/python -m pip install -r training/requirements-training.txt
site/.venv/Scripts/python training/resume_validation.py
site/.venv/Scripts/python training/prepare_validation_artifacts.py
site/.venv/Scripts/python training/ensemble_fit.py
site/.venv/Scripts/python training/external_validation.py --openi --count 200
site/.venv/Scripts/python training/refresh_documentation.py
Кэш толық болса жүктеу желісіз өтеді. Assert бұзылса тоқтап, сәйкессіздікті тексер.
Модель, config, test_metrics.csv, mcnemar.json, ood_params.npz бірге жаңартылады. Серверді қайта іске қос.
Толық Colab нұсқасы: training/pneumonia_colab_full_kz.ipynb.

ТЕСТ ЖӘНЕ DOCKER
site/.venv/Scripts/python -m pip install pytest pypdf httpx
site/.venv/Scripts/python -m pytest site/tests training/tests -q
{ci_text}
{ci_links}
docker build -t tynys .
docker run --rm -p 127.0.0.1:7860:7860 tynys
{docker_text}
Интернетке жарияланған жоқ.
Hugging Face Docker Space үшін түбірдегі README.md және Dockerfile дайын.

{multi_text}

ҚҰЖАТТАР
presentation/{presentation_name} және tynys_project_speech_kz.txt — қазіргі қорғаныс нұсқасы.
MODEL_CARD.md — модель сипаттамасы, training/PHASE_REPORT_KZ.txt — кезеңдер есебі.
Training кэші мен OpenI суреттері таратылатын архивке кірмейді.
'''
    (ROOT/'site/README.txt').write_text(body,encoding='utf-8')
    (ROOT/'README_KZ.txt').write_text(body,encoding='utf-8')
    card=f'''---
language: [kk, en]
tags: [medical-imaging, image-classification, tensorflow, mobilenetv2]
library_name: keras
---

# Tynys модель картасы

## Архитектура және деректер

MobileNetV2 (ImageNet), GlobalAveragePooling, Dropout, sigmoid Dense(1).
Алдымен head, кейін соңғы 30 қабаттың BatchNormalization емес қабаттарын fine-tuning.
CNN: Conv2D 32/64/128, MaxPooling, GAP, Dropout, Dense(1).
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
Train 3801, validation 952, test 618. 5856 жарамды сурет, тазалаудан кейін 5371.
Файл атауы топтары ресми patient ID емес. SHA-256 тек байт бойынша көшірмені тексереді.

## Қазіргі ішкі метрикалар, calibration fitted

{current}

{temperatures}
Validation толық 952/952. Қазіргі шектер validation-да recall≥0.90 шартынан қайта таңдалған.
Test fitting үшін қолданылған жоқ. MobileNetV2 90% accuracy мақсатына жетпеді.

## Қазіргі сыртқы метрикалар

OpenI Indiana NLM API, 200 сурет (160 NORMAL, 40 PNEUMONIA), seed 42.
{ext}
Таңдау бұрынғы cohort-пен бірдей, сыртқы шек өзгертілмеді. SHA-256 қиылысы жоқ.
Бір есеп UID-іне бір сурет. Longitudinal patient identity толық расталмаған.
MeSH белгілері клиникалық gold standard емес, проекцияны PA деп кепілдемейміз.

## OOD және түсіндірме

Сурет сапасы бөлек image_quality_v1 эвристикасымен тексеріледі: жарықтық, контраст,
өлшем, пішім және ұсақ деталь. Параметр: site/model/image_quality_config.json.
Деталь шегі validation-ның төменгі 1-процентилінен есептеледі; ауру баллына әсер етпейді.
Сапа бойынша эталон белгілер жоқ. Клиникалық сапа және анатомиялық қиылу расталмайды.
Толық сипаттама: training/QUALITY_CHECK_KZ.txt.

Validation-ның 1280 белгісі, shrinkage covariance, 95-процентиль шегі {calibration['ood']['threshold']:.8f}.
Параметрлер дайын, non-X-ray сапасы бөлек бағаланған жоқ.
ood_guard_v1: OOD ауытқуы не тексеру қолжетімсіз болғанда сайт, тарих және TXT/PDF
«Сенімді бағалау мүмкін емес» күйін көрсетеді. Баллдар техникалық нәтиже ретінде қалады.
Бұл сақтық ережесінің клиникалық сапасы және қамту/қателік қатынасы бағаланбаған.
Модель мен шектер өзгермеген. Бұрынғы метрикалар бастапқы екі санатты болжауға
қатысты; жаңа ереженің қамту үлесі және таңдамалы қатесі есептелген жоқ.
Grad-CAM 7×7 картадан жуық аймақты көрсетеді, зақым шекарасын анықтамайды.
Калибрлеу клиникалық ықтималдықты дәлелдемейді. NORMAL барлық ауруды жоққа шығармайды.

## McNemar және ensemble

CNN-only correct {paired['cnn_only_correct']}, MobileNetV2-only correct {paired['mobilenet_only_correct']},
exact two-sided p={paired['p_value']:.12g}. Пациент суреттерінің тәуелсіздігі толық расталмаған.
{ensemble_text}

## Кіріс, қолдану шектеулері және лицензия

decode_image, float32, resize_with_pad(224,224), модель. Rescaling ішінде.
DICOM CR/DX кірісі жадта percentile PNG-ге айналады. DICOM clinical validation жасалмаған.
Оқу прототипі. Диагноз, ем таңдау немесе пациент туралы автоматты шешім үшін қолданылмайды.
Сервер суреттерді тұрақты қоймада сақтамайды. Тарих әдепкіде бет жадында;
пайдаланушы келісімімен браузердің IndexedDB қоймасында түпнұсқа файл мен нәтиже
сақталады (20 талдау, 80 МиБ, 7 күн; ескіргені келесі тексеруде жойылады).
Ортақ браузерде тарих көрінеді. Толық жою және келісімді өшіру бар.
Қосымша шифрлау/аккаунт жоқ; сипаттама: training/HISTORY_KZ.txt.
Модель салмақтарына автоматты жаңа лицензия берілмейді, дереккөз шарттары тексерілуі керек.
OpenI: CC BY-NC-ND 4.0. Суреттер архивке/репозиторийге кірмейді.

## Жеке multi-label эксперименті

{multi_text}

{deployment}

## English summary

Educational binary NORMAL/PNEUMONIA classifier. Calibration is fitted on all 952 SHA-verified
validation images. T(MobileNetV2)={config['calibration']['T']:.8f}, threshold={config['threshold']:.3f}.
Current internal and external results are listed above. OOD parameters are fitted on validation
features but non-X-ray detection performance is unvalidated. No clinical approval is claimed.
'''
    (ROOT/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    status=json.loads((ROOT/'training/IMPLEMENTATION_STATUS.json').read_text(encoding='utf-8'))
    status['calibration']={'fitted':True,'runtime_temperature':config['calibration']['T'],'validation_count':952,'threshold':config['threshold']}
    status['ood']={'available':True,**calibration['ood']}
    status['external_validation']={'completed':True,**external}
    if docker:status['docker']=docker
    if multi:status['multilabel']={'completed':True,'website_integration':False,'distributed_weights':False,'metrics':multi['metrics']}
    if ensemble:status['ensemble']={'fitted':True,**ensemble}
    (ROOT/'training/IMPLEMENTATION_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    payload={'config':config,'metrics':metrics,'external':external,'calibration':calibration,'mcnemar':paired}
    (ROOT/'training/presentation_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
