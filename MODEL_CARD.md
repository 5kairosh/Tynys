---
language: [kk, en]
tags: [medical-imaging, image-classification, tensorflow, mobilenetv2]
library_name: keras
---

# Tynys модель картасы

## Міндеті және архитектура

Екі санатты оқу классификаторы: NORMAL (0), PNEUMONIA (1).
MobileNetV2 ImageNet салмақтарымен, GlobalAveragePooling, Dropout және sigmoid Dense(1).
Бастапқы классификаторды үйреткеннен кейін соңғы 30 қабаттың BatchNormalization емес қабаттарын баптау.
CNN baseline: Conv2D 32/64/128, MaxPooling, GlobalAveragePooling, Dropout, Dense(1).

Кіріс: `decode_image(channels=3, expand_animations=False)`, float32, `resize_with_pad(224,224)`.
Rescaling модель ішінде. EXIF айналдыруы, сыртқы нормализация, crop қолданылмайды.
DICOM CR/DX кірісі алдымен 1–99 процентиль, Modality LUT және MONOCHROME1 инверсиясы арқылы жадта PNG-ге айналады.
DICOM осы деректерде бөлек клиникалық валидациядан өтпеген.

## Деректер

[Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
5856 жарамды сурет, тазалаудан кейін 5371. Train 3801, validation 952, test 618.
SHA-256 тек байт бойынша көшірмені тексереді. Файл атауынан алынған топтар ресми пациент ID емес.

## Ішкі бағалау

Сақталған бастапқы модель, 618 test суреті: accuracy 0.891586, precision 0.877358,
recall 0.961240, specificity 0.774892, F1 0.917386, ROC-AUC 0.958075.
Шек 0.740, тек validation-да recall >= 0.90 шартынан specificity, F1 арқылы таңдалған.
TN 179, FP 52, FN 15, TP 372. 90% accuracy мақсатына жетпеді.
CNN accuracy 0.716828, шек 0.485.
Жұптық exact McNemar: CNN-only correct 20, MobileNetV2-only correct 128,
p = 1.8041385810152942e-20. Бір пациенттің суреттері тәуелсіз болмауы мүмкін.

## Сыртқы бағалау және калибрлеу

Орындалу күйі мен нақты нәтижелері `training/external_validation_report.txt` және
`training/external_validation_metrics.csv` ішінде, бар болса беріледі.
Калибрлеу T және шек `site/model/model_config.json` ішінде сақталады.
`calibration` кілті болмаса T=1, бастапқы режим. Validation дерегінен есептемей T ойдан қосылмайды.
Temperature scaling сыртқы клиникалық калибрлеуді дәлелдемейді.

## OOD және түсіндірме

OOD validation-ның 1280 өлшемді белгілері, shrinkage covariance және Махаланобис қашықтығының
95-процентиліне сүйенеді. Бұл non-X-ray деректерде дәлелденген детектор емес.
Параметрлер жоқ/жарамсыз болса ескерту қолжетімсіз, негізгі болжам жұмыс істейді.
Grad-CAM 7×7 картадан жуық назар аймағын береді, зақым шекарасын анықтамайды.

## Қолдану шектеулері

Оқу, модельді салыстыру және бастапқы сұрыптау мүмкіндігін зерттеу.
Клиникалық диагноз, ем таңдау немесе пациентті автоматты өткізу үшін қолданылмайды.
Сыртқы популяциядағы сапа, балалар/ересектер айырмасы, аппарат пен проекция әсері бөлек тексеріледі.
NORMAL барлық ауруды жоққа шығармайды. Балл клиникалық сенімділік пайызы емес.
Жүктелген суреттер серверде және браузердің тұрақты қоймасында сақталмайды. Тарих тек бет жадында.

## Лицензия

Жоба салмақтарына автоматты түрде жаңа лицензия берілмейді.
Датасет авторларының, ImageNet/MobileNetV2 және бастапқы файлдардың шарттарын тексеру қажет.
OpenI Indiana материалдары: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/),
тек сыртқы бағалау, архивке суреттерді қоспау. Есептегі MeSH белгілері gold standard емес.

## English summary

Educational binary chest X-ray classifier (NORMAL/PNEUMONIA), using a fine-tuned MobileNetV2
and a CNN baseline. Internal metrics and exact paired McNemar results are reported above.
No clinical diagnostic approval is claimed. Temperature scaling and Mahalanobis OOD warnings
require validation-derived artifacts. Missing optional artifacts do not block inference.
See the external evaluation report for completed external measurements and cohort limitations.
