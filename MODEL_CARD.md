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

MobileNetV2: T=0.97995943, validation NLL 0.08822560 → 0.08820212.
CNN: T=0.68202770, validation NLL 0.29725222 → 0.27995523.
Validation толық 952/952. Қазіргі шектер validation-да recall≥0.90 шартынан қайта таңдалған.
Test fitting үшін қолданылған жоқ. MobileNetV2 90% accuracy мақсатына жетпеді.

## Қазіргі сыртқы метрикалар

OpenI Indiana NLM API, 200 сурет (160 NORMAL, 40 PNEUMONIA), seed 42.
MobileNetV2: шек 0.745, accuracy 55.00%, precision 23.96%, recall 57.50%, specificity 54.37%, F1 0.3382, ROC-AUC 0.5442; TN 87, FP 73, FN 17, TP 23.
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
Ensemble әлі қосылған жоқ, бұл келесі кезең.

## Кіріс, қолдану шектеулері және лицензия

decode_image, float32, resize_with_pad(224,224), модель. Rescaling ішінде.
DICOM CR/DX кірісі жадта percentile PNG-ге айналады. DICOM clinical validation жасалмаған.
Оқу прототипі. Диагноз, ем таңдау немесе пациент туралы автоматты шешім үшін қолданылмайды.
Сайт суреттерді тұрақты қоймада сақтамайды. Тарих тек бет жадында.
Модель салмақтарына автоматты жаңа лицензия берілмейді, дереккөз шарттары тексерілуі керек.
OpenI: CC BY-NC-ND 4.0. Суреттер архивке/репозиторийге кірмейді.

## English summary

Educational binary NORMAL/PNEUMONIA classifier. Calibration is fitted on all 952 SHA-verified
validation images. T(MobileNetV2)=0.97995943, threshold=0.745.
Current internal and external results are listed above. OOD parameters are fitted on validation
features but non-X-ray detection performance is unvalidated. No clinical approval is claimed.
