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

Толық нұсқаулық: `README_KZ.txt`. Модель сипаттамасы: `MODEL_CARD.md`.

Hugging Face Docker Space-қа осы репозиторий мазмұнын салыңыз. Түбірдегі Dockerfile қолданылады.
Жергілікті іске қосу: `site/start.bat`. Интернетке жариялау бұл тапсырмада жасалған жоқ.

Docker тексеруі: `docker build -t tynys .`, кейін `docker run --rm -p 127.0.0.1:7860:7860 tynys`.

Датасет лицензиясы мен модель салмақтарын тарату шарттарын жариялау алдында тексеру қажет.

Docker: build/run тексерілді (image tynys, Server 29.8.1). Image 3.28GB. Жергілікті тексеру порты: 7861. Дәлелдер: `training/docker_evidence/`.

## GitHub және CI

[Репозиторий](https://github.com/5kairosh/Tynys)

[![CI](https://github.com/5kairosh/Tynys/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/5kairosh/Tynys/actions/workflows/ci.yml)

CI қашықта өтті (ubuntu-latest, Python 3.11, 56 тест).
[Расталған run](https://github.com/5kairosh/Tynys/actions/runs/38745485393).
Дәлелдер: `training/ci_evidence/success_run.json`, `pytest_success.txt`.
