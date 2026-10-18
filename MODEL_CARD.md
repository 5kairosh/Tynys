---
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

CNN: шек 0.475, accuracy 71.52%, precision 70.81%, recall 92.76%, specificity 35.93%, F1 0.8031, ROC-AUC 0.8211; TN 83, FP 148, FN 28, TP 359.
MobileNetV2: шек 0.745, accuracy 89.16%, precision 87.74%, recall 96.12%, specificity 77.49%, F1 0.9174, ROC-AUC 0.9581; TN 179, FP 52, FN 15, TP 372.
Ensemble: шек 0.745, accuracy 89.16%, precision 87.74%, recall 96.12%, specificity 77.49%, F1 0.9174, ROC-AUC 0.9583; TN 179, FP 52, FN 15, TP 372.

MobileNetV2: T=0.97995943, validation NLL 0.08822560 → 0.08820212.
CNN: T=0.68202770, validation NLL 0.29725222 → 0.27995523.
Validation толық 952/952. Қазіргі шектер validation-да recall≥0.90 шартынан қайта таңдалған.
Test fitting үшін қолданылған жоқ. MobileNetV2 90% accuracy мақсатына жетпеді.

## Қазіргі сыртқы метрикалар

OpenI Indiana NLM API, 200 сурет (160 NORMAL, 40 PNEUMONIA), seed 42.
CNN: шек 0.475, accuracy 21.50%, precision 20.30%, recall 100.00%, specificity 1.88%, F1 0.3376, ROC-AUC 0.4706; TN 3, FP 157, FN 0, TP 40.
MobileNetV2: шек 0.745, accuracy 55.00%, precision 23.96%, recall 57.50%, specificity 54.37%, F1 0.3382, ROC-AUC 0.5442; TN 87, FP 73, FN 17, TP 23.
Ensemble: шек 0.745, accuracy 55.00%, precision 23.96%, recall 57.50%, specificity 54.37%, F1 0.3382, ROC-AUC 0.5441; TN 87, FP 73, FN 17, TP 23.
Таңдау бұрынғы cohort-пен бірдей, сыртқы шек өзгертілмеді. SHA-256 қиылысы жоқ.
Бір есеп UID-іне бір сурет. Longitudinal patient identity толық расталмаған.
MeSH белгілері клиникалық gold standard емес, проекцияны PA деп кепілдемейміз.

## OOD және түсіндірме

Validation-ның 1280 белгісі, shrinkage covariance, 95-процентиль шегі 26.34090290.
Параметрлер дайын, non-X-ray сапасы бөлек бағаланған жоқ. OOD тек ескерту береді.
Grad-CAM 7×7 картадан жуық аймақты көрсетеді, зақым шекарасын анықтамайды.
Калибрлеу клиникалық ықтималдықты дәлелдемейді. NORMAL барлық ауруды жоққа шығармайды.

## McNemar және ensemble

CNN-only correct 19, MobileNetV2-only correct 128,
exact two-sided p=4.8323820233e-21. Пациент суреттерінің тәуелсіздігі толық расталмаған.
Ensemble: validation-да таңдалған салмақтар {"CNN": 0.001005948939264576, "MobileNetV2": 0.9989940510607355}, шек 0.745. Сыртқы белгілер таңдау үшін қолданылған жоқ.

## Кіріс, қолдану шектеулері және лицензия

decode_image, float32, resize_with_pad(224,224), модель. Rescaling ішінде.
DICOM CR/DX кірісі жадта percentile PNG-ге айналады. DICOM clinical validation жасалмаған.
Оқу прототипі. Диагноз, ем таңдау немесе пациент туралы автоматты шешім үшін қолданылмайды.
Сайт суреттерді тұрақты қоймада сақтамайды. Тарих тек бет жадында.
Модель салмақтарына автоматты жаңа лицензия берілмейді, дереккөз шарттары тексерілуі керек.
OpenI: CC BY-NC-ND 4.0. Суреттер архивке/репозиторийге кірмейді.

## Жеке multi-label эксперименті

MULTI-LABEL — ЖЕКЕ ОҚУ ЭКСПЕРИМЕНТІ
{"train": {"images": 434, "groups": 233, "positive_counts": {"pneumonia": 54, "atelectasis": 151, "cardiomegaly": 141}}, "validation": {"images": 91, "groups": 49, "positive_counts": {"pneumonia": 10, "atelectasis": 30, "cardiomegaly": 28}}, "test": {"images": 92, "groups": 49, "positive_counts": {"pneumonia": 10, "atelectasis": 29, "cardiomegaly": 29}}}
pneumonia: шек 0.26, AUC 0.5378, F1 0.2308, sensitivity 30.00%, specificity 84.15%; TN 69, FP 13, FN 7, TP 3.
atelectasis: шек 0.17, AUC 0.6196, F1 0.4828, sensitivity 96.55%, specificity 6.35%; TN 4, FP 59, FN 1, TP 28.
cardiomegaly: шек 0.18, AUC 0.5632, F1 0.4894, sensitivity 79.31%, specificity 33.33%; TN 21, FP 42, FN 6, TP 23.
ImageNet MobileNetV2 + GAP + үш sigmoid Dense(1), BCE. Соңғы 30 қабат, BN мұздатылған. Таңдалған кезең: fine.
Шектер тек validation F1 арқылы. MeSH report-level әлсіз белгілер, жоқ белгі weak negative деп алынған.
Report ID longitudinal пациент ID емес; frontal/lateral аралас. Шағын квоталық жиын жалпы популяцияны көрсетпейді.
Binary сыртқы cohort-пен сурет қиылысы: 71; бұл жаңа тәуелсіз binary тексеру емес.
OpenI суреттерін, үйретілген салмақтарын және қате суреттерін таратуға болмайды: олар тек жергілікті private/cache бумасында, Git/ZIP-ке кірмейді.
Сайтқа қосылған жоқ. Қайта орындау: training/multilabel_colab_kz.ipynb немесе training/multilabel_experiment.py.
Толық сандар: training/outputs/multilabel/per_label_metrics.csv және experiment_report.json.


## Deployment

Docker: build/run тексерілді (image tynys, Server 29.8.1). Image көлемі: 3.28GB (3281444328 байт). Тексерілген host порт: 7861, контейнер: 7860. PNG/CR DICOM 200; CT DICOM/мәтінді JPG 400; PDF 200, 2 бет; Traceback жоқ. Контейнер жойылды.

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
   ood_params.npz, mcnemar.json және external_validation.json бірге болуы керек.
4. Build біткен соң Space Logs пен `/api/status` тексеріңіз. Контейнер 0.0.0.0:7860 тыңдайды.
   Venv, cache, пайдаланушы суреттері, OpenI суреттері мен multi-label салмақтарын жүктемеңіз.

Space интернетке осы тапсырмада жарияланған жоқ. Ресми нұсқаулық:
[Hugging Face Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).


## English summary

### OOD display guard (2026-10-18)

`ood_guard_v1` withholds the headline NORMAL/PNEUMONIA classification when the existing
distribution check flags an input or is unavailable/incomplete. The UI, session history,
TXT and PDF show «Сенімді бағалау мүмкін емес»; scores and model classes remain available
as technical outputs. Grad-CAM is optional, with the original image shown by default
for these inputs. The API adds `assessment.status` (`classified` or `review_required`),
`reason`, `policy` and `clinically_validated: false`; existing model scores/classes remain unchanged.

This is a conservative presentation rule, not a validated selective classifier, an X-ray
authenticator or a clinical confidence estimate. It does not change model weights, calibration,
the 0.745 decision threshold or the validation OOD threshold. Existing internal/external
metrics measure the original forced binary predictions; coverage and selective error rates
for this guard have not been evaluated. A passed OOD check does not establish correctness.

Educational binary NORMAL/PNEUMONIA classifier. Calibration is fitted on all 952 SHA-verified
validation images. T(MobileNetV2)=0.97995943, threshold=0.745.
Current internal and external results are listed above. OOD parameters are fitted on validation
features but non-X-ray detection performance is unvalidated. No clinical approval is claimed.
