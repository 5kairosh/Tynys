---
language: [kk, en]
tags: [medical-imaging, image-classification, tensorflow, mobilenetv2]
library_name: keras
---

# Tynys модель картасы — 6-кезең, 2026-10-21

Оқу зерттеу прототипі. NORMAL/PNEUMONIA санаттары клиникалық диагноз емес;
NORMAL барлық ауруды жоққа шығармайды. Балл ауру ықтималдығы немесе сенімділік пайызы емес.

## Архитектура және таңдау

Frozen MobileNetV2 (ImageNet, бұрын Kaggle-да fine-tuning), GAP1280, екі frozen Dense
expert, source-domain logistic router C0.001. Kaggle бастапқы expert сақталған,
RSNA expert жаңа train-де үйретілген. Тармақ source-logit>=0 ережесімен таңдалады.
Әр expert operating margin-і common threshold0.5-ке келтірілген; соңғы T1.1374971593841223
тек validation NLL-мен fit болған. Барлығы стандартты Keras layers/ops .keras ішінде;
safe_mode=True, custom objects/Lambda жоқ. Router жас/диагноз/дұрыстықты анықтамайды.
CNN бастапқы салмағы мен T0.6820277027003551 сақталған; ensemble1352 validation-да қайта fit.
Модель SHA256: a79fe71a67fc4e100a5cbde8624fb9286e663bd59e4beb74984810b3f2ba4e43.

## Деректер және жаңа тексеру

Kaggle3801 train/952 validation/618 test; RSNA1600 train/400 validation/400 prior
holdout/400 new confirmation. NIH mapping пациенттері split-терге қайталанбайды.
Барлық2800 RSNA пациентінен бір сурет, source ZIP CRC және PNG SHA тексерілген.
Calculated Normal vs Lung Opacity ғана; басқа abnormal findings NORMAL емес.
Оң RSNA белгісі — пневмонияға күдікті күңгірттену, клиникалық расталған пневмония емес.
RSNA үйретуге қатысқан дереккөз: confirmation жаңа пациенттерде, жаңа мекемеде емес.
OpenI200 тек evaluation, әлсіз report MeSH. Kaggle/OpenI және алғашқы RSNA400 бұрын
қаралған; тек соңғы400 RSNA пациенті 6B дизайны үшін жаңа болды. Қайта tuning жоқ.
6A ішкі balanced accuracy талабынан өтпей енгізілмеді; 6B екінші жаңа holdout алды.

## Қазіргі барлық суретке техникалық болжам нәтижесі

- RSNA жаңа400: accuracy88%, balanced accuracy88%, sensitivity84%, specificity92%,
  AUC0.937925; TN184 FP16 FN32 TP168. Бастапқы68%, қате128→48.
  Жұптық bootstrap gain95% аралығы +15…+25 пайыздық тармақ (10000 resamples).
- OpenI200: accuracy64.5%, balanced accuracy67.5%, sensitivity72.5%, specificity62.5%,
  AUC0.755; TN100 FP60 FN11 TP29. Бастапқы55%, қате90→71. Қайта тексеру ғана.
- Kaggle618: accuracy88.9968%, sensitivity96.124%, specificity77.056%, AUC0.957806;
  TN178 FP53 FN15 TP372. Бұрынғы89.1586%, қате67→68. Ішкі сәл төмендеу жасырылмаған.

RSNA квотасы50/50: accuracy клиникадағы prevalence-ті бағаламайды. Мекемелер,
құрылғылар және популяциялар бойынша клиникалық сапа расталмаған. OpenI қатесі әлі35.5%.

## OOD, қамту және түсіндірме

Train5401 reference / бөлек validation1352 cutoff; covariance shrinkage0.1,
95% target, rank1286, cutoff38.37188343647505. Formal coverage guarantee жоқ.
Evaluation суреттері threshold fit-ке кірмейді. OOD X-ray authentication емес.
Қорытынды: Kaggle608/618, қате68/608; OpenI167/200, қате53/167;
RSNA348/400, қате45/348. Бұл denominator all-cohort accuracy-мен бірдей емес.
Ескі OpenI accepted32 ішіндегі20 қате орнына енді167 ішіндегі53 қате бар:
абсолют қате қорытынды азайды немесе клиникалық қауіпсіздік артты деуге болмайды.

Grad-CAM дәл таңдалған expert margin-інің градиентіне есептеледі. Ол router таңдауының
себебін түсіндірмейді; ауру ошағы/шекарасы/диагноз дәлелі емес. Техникалық image-quality
warnings диагнозды, анатомиялық қамтуды немесе X-ray түпнұсқалығын растамайды.
Тармақ сайт/TXT/PDF-те көрсетіледі. Ескі opt-in history қайта есептелмейді.

## Дәлел және атрибуция

150 pytest +29 Chromium/IndexedDB жергілікті тесті өтті. Нақты Kaggle/RSNA source-та
толық API паритеті және екі expert Grad-CAM reconstruction-ы тексерілді. PDF2бет визуалды тексерілді.
CI/онлайн үшін training/generalization_stage6/verification.json соңғы күйін қараңыз.
Әдіс/протокол/шектеу: training/GENERALIZATION_STAGE6_KZ.txt.
Source terms/citations: training/generalization_stage6/DATA_ATTRIBUTION.md.
NIH Clinical Center; Wang et al., CVPR2017; RSNA/STR; Shih et al., Radiology AI2019,
https://doi.org/10.1148/ryai.2019180041 . OpenI evaluation-only, оның суреті/салмағы таратылмайды.
Сурет/GAP/DICOM кэштері, private және пайдаланушы суреттері Git/ZIP-ке кірмейді.
Бастапқы модель картасы: training/generalization_stage6/baseline_MODEL_CARD.md (тарихи).
Бастапқы Colab notebook және презентация6-кезең моделін қамтымайды.

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
   ood_params.npz, mcnemar.json, external_validation.json және image_quality_config.json бірге болуы керек.
4. Build біткен соң Space Logs пен `/api/status` тексеріңіз. Контейнер 0.0.0.0:7860 тыңдайды.
   Venv, cache, пайдаланушы суреттері, OpenI суреттері мен multi-label салмақтарын жүктемеңіз.

Space интернетке осы тапсырмада жарияланған жоқ. Ресми нұсқаулық:
[Hugging Face Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).



2026-10-21: жоғарыдағы Docker тексеруі тарихи. 6-кезең image build/run қайта тексерілмеді.

6-кезең CI және онлайн тексерілді: [CI run](https://github.com/5kairosh/Tynys/actions/runs/39489492345). Дәлел: `training/generalization_stage6/verification.json`.


## Stage 7 service limits (2026-10-21)

The clinical/model scope is unchanged. One inference worker shares three total
admission slots across uploads, queued jobs, predictions and PDF jobs. Excess
jobs receive HTTP 429 before body buffering. Queue and response deadlines do not
interrupt native TensorFlow execution: detached work retains its slot until it
finishes. `/api/health` is liveness; `/api/ready` checks that both models loaded.
Neither guarantees prediction quality, free capacity, or continuous hosting.
The browser never automatically re-uploads an image after a failure.
See `training/SERVICE_STAGE7_KZ.txt` and the short synthetic load evidence under
`training/service_stage7/`. Codespaces can still sleep. No current Docker rebuild
or sustained capacity/24/7 uptime claim is made.
