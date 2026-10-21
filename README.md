---
title: Tynys
emoji: 🫁
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Tynys

## Public website

[Open Tynys](https://kairosh--tynys.us-east.modal.run/)

Deployed to Modal on 2026-10-21. The public API was verified with PNG/JPG,
CR DICOM, Grad-CAM, three-model comparison, OOD, and a two-page PDF report.
The deployment is independent of the local computer and uses automatic cold starts.
This cardless Starter account currently has a $1 free usage cap; no payment
method was added. Quota exhaustion can interrupt access, so this is not an
unlimited availability guarantee. [Deployment guide](deployment/README_MODAL_KZ.txt).

Stage 8 (2026-10-21): optional password-encrypted browser history, manual and
five-minute idle locks, and suppression of late predictions/PDFs after locking.
Legacy history is preserved until the user elects protection. Passwords and keys
are not sent to the API; there is no password recovery. Downloaded files remain
outside the vault. [Guide and limits](training/HISTORY_VAULT_STAGE8_KZ.txt).
[Verified CI: 159 Python + 48 Chromium/Web Crypto tests](https://github.com/5kairosh/Tynys/actions/runs/39498827595).
The medical inference pipeline is unchanged.

Stage 7 (2026-10-21): bounded inference admission (3 total slots, one worker),
shared PNG/DICOM/PDF backpressure, upload/queue/response deadlines, health/readiness,
and browser recovery without automatic image re-upload. Local short burst: 3 HTTP 200,
5 HTTP 429, health p95 15.81 ms. 159 Python and 33 Chromium tests passed.
No model/OOD changes; 1218-image forward audit reproduced previous predictions.
[Stage 7 CI passed: 159 Python + 33 browser tests](https://github.com/5kairosh/Tynys/actions/runs/39494920388).
See [Kazakh service guide](training/SERVICE_STAGE7_KZ.txt).
Codespaces remains a temporary demo, with no 24/7 availability guarantee.

6-КЕЗЕҢ — ҚАЗІРГІ МОДЕЛЬ, 2026-10-21
Екі үйретілген тармақты MobileNetV2: Kaggle бастапқы + RSNA бейімделген, source router.
Train5401 / validation1352; жаңа RSNA confirmation400 пациенті бөлек.
Шешім шегі0.5; T1.1374971593841223. Model ID a79fe71a67fc, OOD ID b0c5e869838e.
RSNA жаңа400: accuracy68→88%, қате128→48; OpenI200 қайта тексеру55→64.5%, қате90→71.
Ішкі618: accuracy89.16→89.00%, қате67→68. Барлық деректе жақсарды деген сөз емес.
RSNA-дағы мақсат — өкпе күңгірттенуі; клиникалық пневмония диагнозы емес.
RSNA үйретуге де қолданылды: жаңа пациенттер тексерілді, жаңа мекеме емес.
OpenI мен Kaggle бұрын қаралған. Модель клиникалық қолдануға мақұлданбаған.
OOD accepted: Kaggle608/618 (68қате), OpenI167/200 (53қате), RSNA348/400 (45қате).
Қамту, accepted-қате және барлық сурет accuracy-і бөлек көрсетіледі.
Толық әдіс/шектеу: training/GENERALIZATION_STAGE6_KZ.txt.
Дереккөз: training/generalization_stage6/DATA_ATTRIBUTION.md.
Презентация/бастапқы Colab notebook — тарихи бастапқы нұсқа; 6-кезеңді қамтымайды.
Docker осы кезеңде қайта жиналмады. Codespaces уақытша орта, 24/7 кепілдік жоқ.
Төмендегі алдыңғы кезең жазбалары — тарихи нәтижелер.


Өкпе рентгенін NORMAL/PNEUMONIA санаттарына жіктейтін қазақша оқу прототипі.
Бұл медициналық диагноз емес. NORMAL барлық ауруды жоққа шығармайды.

CNN және MobileNetV2 салыстыруы, Grad-CAM, PDF есеп, сессия тарихы және CR/DX DICOM қолдауы бар.
Калибрлеу мен OOD тек сәйкес validation артефактілері бар кезде қолданылады.

OOD ауытқуы байқалса немесе тексеру орындалмаса, нәтиже тақырыбы «Сенімді бағалау мүмкін емес» болады.
Бұл күй тарих пен TXT/PDF есептерде де сақталады; модель баллдары тек техникалық нәтиже ретінде көрсетіледі.
Бұл OOD-қа сүйенетін сақтық ережесі. Оның қамту/қателік аудиті төменде берілген; клиникалық сапасы расталмаған.
Осы өзгеріс: жергілікті 65 тест өтті; [GitHub CI сәтті](https://github.com/5kairosh/Tynys/actions/runs/39125943690). Тексерілген код: `eed3b495a684a09a883235ed4d152b82bc3ed999`.
Онлайн тексеру дәлелі: `training/guard_evidence/verification.json`.

Толық нұсқаулық: `README_KZ.txt`. Модель сипаттамасы: `MODEL_CARD.md`.

Тарихты осы браузерде сақтау — тек пайдаланушы келісімімен. Әдепкіде өшірулі;
қосылса соңғы 20 талдау (түпнұсқа файл + нәтиже), 80 МиБ, 7 күндік мерзім.
Ескіргендері сайт ашылғанда/белсенді тексеруде жойылады. Сақтауды өшіру немесе
«Тарихты тазалау» сақталған көшірмелерді жояды. Серверде тарих көшірмесі жасалмайды.
Ортақ браузер профилін пайдаланатын адамдар тарихты көре алады. Нұсқау: `training/HISTORY_KZ.txt`.
Тексеру: 89 серверлік тест және 16 Chromium/IndexedDB сынағы өтті. [GitHub CI сәтті](https://github.com/5kairosh/Tynys/actions/runs/39197545496), code commit `f423420a308cd6dbf81eaec092af0109e515aacd`.
Онлайн тексеру дәлелі: `training/history_evidence/verification.json`.

Сурет сапасы бөлек тексеріледі: шағын өлшем, төмен контраст, тым қараңғы/жарық пиксельдер,
әдеттен тыс пішім және ұсақ детальдың аздығы туралы ескертулер беріледі.
Деталь шегі 952 SHA256-ы тексерілген validation суретінің жарамды 947 өлшемінен
төменгі 1-процентиль арқылы есептелген. Ескертулер сайтта, тарихта және TXT/PDF-те көрсетіледі.
Бұл тексеру модель баллын өзгертпейді және рентгенді, анатомиялық қиылуды не клиникалық сапаны растамайды.
Сипаттама: `training/QUALITY_CHECK_KZ.txt`; параметр: `site/model/image_quality_config.json`.
Жергілікті 89 тест өтті. [Осы кодтың CI тексеруі сәтті](https://github.com/5kairosh/Tynys/actions/runs/39127904736) (code commit `a86a02c53774e631fc37c3bc13b69ee52c3d62b3`).
Онлайн PNG/DICOM/PDF тексеру дәлелі: `training/quality_evidence/verification.json`.

Hugging Face Docker Space-қа осы репозиторий мазмұнын салыңыз. Түбірдегі Dockerfile қолданылады.
Жергілікті іске қосу: `site/start.bat`. Hugging Face Space-қа жарияланған жоқ; Codespaces демонстрациясы төменде.

Docker тексеруі: `docker build -t tynys .`, кейін `docker run --rm -p 127.0.0.1:7860:7860 tynys`.

Датасет лицензиясы мен модель салмақтарын тарату шарттарын жариялау алдында тексеру қажет.

Docker: build/run тексерілді (image tynys, Server 29.8.1). Image 3.28GB. Жергілікті тексеру порты: 7861. Дәлелдер: `training/docker_evidence/`.

## GitHub және CI

[Репозиторий](https://github.com/5kairosh/Tynys)

[![CI](https://github.com/5kairosh/Tynys/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/5kairosh/Tynys/actions/workflows/ci.yml)

Бастапқы CI: ubuntu-latest, Python 3.11, 56 тест. Қазіргі тексеру: 135 pytest + 24 Chromium; 5-кезең жазбасы төменде.
[Расталған run](https://github.com/5kairosh/Tynys/actions/runs/38745485393).
Дәлелдер: `training/ci_evidence/success_run.json`, `pytest_success.txt`.


## GitHub Codespaces демонстрациясы

[Сайтты ашу](https://effective-space-winner-q74xxqj555vj39qq9-7860.app.github.dev)

2026-10-16: 2 CPU / 8 GB ортада ашық порт арқылы PNG/JPG/CR DICOM, Grad-CAM, үш модельді салыстыру, PNG/TXT/PDF жүктеу, бірнеше файл және тарих тексерілді.
Қаралған код: `d5b38c457f528e16e908e367889711bfbfc8c201`. Дәлел: `training/codespaces_evidence/verification.json`.
Бұл қорғауға арналған уақытша орта: Codespace тоқтағанда сайт та тоқтайды.
[Codespaces басқару](https://github.com/codespaces). Қайта іске қосу: `.devcontainer/README_KZ.txt`.

## 4-КЕЗЕҢ — САҚТЫҚ ЕРЕЖЕСІНІҢ НАҚТЫ ҚАМТУЫ МЕН ҚАТЕЛІГІ
Сайттың «Зерттеу нәтижесі» бөліміне ішкі test 618 / сыртқы OpenI 200 жеке
есептері және JSON жүктеу қосылды. Ішкі қамту 39/618 (6,31%), қабылданған топ
қатесі 6/39 (15,38%); OpenI 0/200, сондықтан таңдамалы accuracy есептелмейді.
Қателікті азайту расталған жоқ. Модель мен шектер өзгермеді.
Әдіс, шектеу, қайта орындау: training/SELECTIVE_EVALUATION_KZ.txt.

4-кезеңнің тексерілген коды: `a4d1fe74662e32a7a89d2b2f2e5f76a0daa6d275`. [CI сәтті](https://github.com/5kairosh/Tynys/actions/runs/39414009947) (екі job). Жергілікті: 109 pytest + 21 Chromium/IndexedDB тесті.

Ашық сайтта PNG/CR DICOM талдауы, екі беттік PDF және дәл сәйкес JSON есеп тексерілді. Дәлелдер: `training/selective_evidence/`. Docker осы кезеңде қайта жиналмады; презентация тарихи нұсқасында қалды.

## 5-КЕЗЕҢ — ҚАЗІРГІ OOD ЕРЕЖЕСІ (2026-10-20)
OOD geometry: train 3801; cutoff: бөлек validation 952, алдын ала 95% target.
OOD шегі 38,25761417386084. Классификация шегі 0,745, модель/T өзгермеді.
Ішкі test қамтуы 39/618 → 598/618 (96,76%), тоқтатылғаны 579 → 20.
Қабылданған топ accuracy 88,96%; барлық test accuracy 89,16% өзгерген жоқ.
OpenI қамтуы 0 → 32/200; сол 32 суреттің 20-сы қате. OOD-тан өту дұрыс
классификацияға кепіл емес. Қалған 168 сыртқы сурет тоқтатылды.
Test/OpenI бұрын қаралған; жаңа тәуелсіз клиникалық тексеру деп саналмайды.
Сипаттама/қайта орындау: training/OOD_STAGE5_KZ.txt.
Сайт ескі тарихқа «OOD ережесі жаңарған» ескертуін береді; ескі санат өзгертілмейді.

## 5-кезеңнің онлайн тексеруі (2026-10-21)

2026-10-21 — 5-КЕЗЕҢДІ ОНЛАЙН ТЕКСЕРУ
Сайт: https://effective-space-winner-q74xxqj555vj39qq9-7860.app.github.dev/
GitHub CI сәтті: https://github.com/5kairosh/Tynys/actions/runs/39481020833
Тексерілген код: 3d9bb94472aba45b99b32eb5b851805c97a396b6; OOD runtime: 02af52e1e8e909cdb041ef64d08e1d4e094cf2a7.
Жергілікті 135 pytest + 24 Chromium/IndexedDB тесті өтті. Кэш түзетілгеннен кейін
24 браузер тесті қайта сәтті өтті (31,87 с); GitHub CI-де екі job да success.
Ашық сервер: synthetic PNG және CR DICOM — HTTP 200; CT DICOM және жалған JPG — 400;
екі файл PDF — HTTP 200, %PDF, 2 бет. Жүктелген JSON /api/status есебімен дәл сәйкес.
Нақты Kaggle рентгендері жергілікті API/браузерде тексерілді; ашық серверге тек
синтетикалық сынақтар жіберілді. Қолданыстағы пайдаланушы тарихы өзгертілмеді.
Браузер енді нұсқаланған CSS/JS алады: ескі кэш жаңа OOD панелін жасырып қалмайды.
Codespace-қа 18 файл SHA256-мен тексеріліп орнатылды; index.html кэш түзетуімен
жаңартылды. Codespace-та git жоқ, Git HEAD жаңарды деп есептелмейді.
Дәлелдер: training/ood_stage5/verification.json, deployment.json, public_api.json,
ci.json, ui_online.jpg. Ескі кезеңдердің жазбалары — тарихи нәтижелер.
Ішкі тест тоқтатуы 579→20; сыртқы OpenI-де 32/200 ғана өтті, оның 20-сы қате.
Бұл түзету классификация дәлдігі немесе клиникалық қауіпсіздік артты деген сөз емес.
Docker осы кезеңде қайта жиналмады. Презентация тарихи нұсқасында қалды.
Codespaces тоқтаса, сайт та тоқтайды; тәулік бойы жұмыс істеу кепілдігі жоқ.

## 6-кезеңнің онлайн тексеруі

2026-10-21 — 6-КЕЗЕҢНІҢ ҚАШЫҚТАҒЫ ТЕКСЕРУІ
Код: eaa6c5aa8e61dfb2d9ff174845994754dc8c256a
CI success, екі job: https://github.com/5kairosh/Tynys/actions/runs/39489492345
Жергілікті150 pytest және29 Chromium/IndexedDB; CI сол suites-ті орындап success болды.
Сайт: https://effective-space-winner-q74xxqj555vj39qq9-7860.app.github.dev/
Жаңа модель a79fe71a67fc, threshold0.5, T1.1374971593841223, OOD b0c5e869838e.
65 файл Codespace-та SHA256-мен тексерілді; Git HEAD жаңартылды деп есептелмейді.
Ашық root/API200; synthetic PNG/CR DICOM200, CT/fakeJPG400; PDF %PDF,2бет.
Браузерден JSON study/қамту есептері жүктеліп, жария evidence/API-мен дәл салыстырылды.
Console error жоқ. Қолданыстағы пайдаланушының браузер тарихы өзгертілмеді.
Resume-тан кейінгі ескі туннель404 берді; loopback API/HTML200 екені дәлелденіп,
7860 forwarding қайта құрылды және Public қалпына келтірілді. Кейін барлық HTTP
және браузер тексерулері қайта өтті. Codespace тоқтаса сайт та тоқтайды; 24/7 кепілдік жоқ.
Дәлел: training/generalization_stage6/verification.json, ci.json, deployment.json,
public_api.json және ui_online.jpg. Docker қайта жиналмады; презентация тарихи.
