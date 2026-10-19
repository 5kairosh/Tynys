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

Өкпе рентгенін NORMAL/PNEUMONIA санаттарына жіктейтін қазақша оқу прототипі.
Бұл медициналық диагноз емес. NORMAL барлық ауруды жоққа шығармайды.

CNN және MobileNetV2 салыстыруы, Grad-CAM, PDF есеп, сессия тарихы және CR/DX DICOM қолдауы бар.
Калибрлеу мен OOD тек сәйкес validation артефактілері бар кезде қолданылады.

OOD ауытқуы байқалса немесе тексеру орындалмаса, нәтиже тақырыбы «Сенімді бағалау мүмкін емес» болады.
Бұл күй тарих пен TXT/PDF есептерде де сақталады; модель баллдары тек техникалық нәтиже ретінде көрсетіледі.
Бұл OOD-қа сүйенетін сақтық ережесі, оның клиникалық сапасы мен қамту/қателік қатынасы әлі бағаланбаған.
Осы өзгеріс: жергілікті 65 тест өтті; [GitHub CI сәтті](https://github.com/5kairosh/Tynys/actions/runs/39125943690). Тексерілген код: `eed3b495a684a09a883235ed4d152b82bc3ed999`.
Онлайн тексеру дәлелі: `training/guard_evidence/verification.json`.

Толық нұсқаулық: `README_KZ.txt`. Модель сипаттамасы: `MODEL_CARD.md`.

Тарихты осы браузерде сақтау — тек пайдаланушы келісімімен. Әдепкіде өшірулі;
қосылса соңғы 20 талдау (түпнұсқа файл + нәтиже), 80 МиБ, 7 күндік мерзім.
Ескіргендері сайт ашылғанда/белсенді тексеруде жойылады. Сақтауды өшіру немесе
«Тарихты тазалау» сақталған көшірмелерді жояды. Серверде тарих көшірмесі жасалмайды.
Ортақ браузер профилін пайдаланатын адамдар тарихты көре алады. Нұсқау: `training/HISTORY_KZ.txt`.

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

CI қашықта өтті (ubuntu-latest, Python 3.11, 56 тест).
[Расталған run](https://github.com/5kairosh/Tynys/actions/runs/38745485393).
Дәлелдер: `training/ci_evidence/success_run.json`, `pytest_success.txt`.


## GitHub Codespaces демонстрациясы

[Сайтты ашу](https://effective-space-winner-q74xxqj555vj39qq9-7860.app.github.dev)

2026-10-16: 2 CPU / 8 GB ортада ашық порт арқылы PNG/JPG/CR DICOM, Grad-CAM, үш модельді салыстыру, PNG/TXT/PDF жүктеу, бірнеше файл және тарих тексерілді.
Қаралған код: `d5b38c457f528e16e908e367889711bfbfc8c201`. Дәлел: `training/codespaces_evidence/verification.json`.
Бұл қорғауға арналған уақытша орта: Codespace тоқтағанда сайт та тоқтайды.
[Codespaces басқару](https://github.com/codespaces). Қайта іске қосу: `.devcontainer/README_KZ.txt`.
