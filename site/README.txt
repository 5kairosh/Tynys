TYNYS — ҚАЗІРГІ ЕСЕПТЕЛГЕН НҰСҚА

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
CNN: шек 0.475, accuracy 71.52%, precision 70.81%, recall 92.76%, specificity 35.93%, F1 0.8031, ROC-AUC 0.8211; TN 83, FP 148, FN 28, TP 359.
MobileNetV2: шек 0.745, accuracy 89.16%, precision 87.74%, recall 96.12%, specificity 77.49%, F1 0.9174, ROC-AUC 0.9581; TN 179, FP 52, FN 15, TP 372.
Ensemble: шек 0.745, accuracy 89.16%, precision 87.74%, recall 96.12%, specificity 77.49%, F1 0.9174, ROC-AUC 0.9583; TN 179, FP 52, FN 15, TP 372.
90% accuracy мақсаты: жетпеді.
Бірдей accuracy қайта есептегенде өзгермеуі мүмкін. Бұл сандар қазіргі конфигурациямен есептелген.

КАЛИБРЛЕУ — ОРЫНДАЛДЫ
952/952 validation файлы SHA-256 бойынша расталды. Test және сыртқы белгілер fitting-ке қатыспайды.
MobileNetV2: T=0.97995943, validation NLL 0.08822560 → 0.08820212.
CNN: T=0.68202770, validation NLL 0.29725222 → 0.27995523.
Әр модельдің баллы sigmoid(logit(raw)/T) арқылы калибрленеді.
Шек тек validation-да recall≥0.90, specificity, F1 ережесімен қайта таңдалған.
Grad-CAM reconstructed_score raw sigmoid-пен тексеріледі, калибрленген баллмен емес.
Калибрлеу клиникалық сенімділікті дәлелдемейді.

OOD — ПАРАМЕТРЛЕРІ ДАЙЫН
Validation: 952, белгілер: 1280, shrinkage covariance.
Махаланобис қашықтығының 95-процентиль шегі: 26.34090290.
site/model/ood_params.npz модель fingerprint-іне байланған.
ood_guard_v1: OOD ауытқуы не тексеру қолжетімсіз болғанда сайт, тарих және TXT/PDF
«Сенімді бағалау мүмкін емес» күйін көрсетеді. Баллдар техникалық нәтиже ретінде қалады.
Ереженің test/OpenI қамтуы мен қателігі өлшенді: training/SELECTIVE_EVALUATION_KZ.txt.
Клиникалық сапасы расталмаған; ішкі қамту 39/618, OpenI қамту 0/200.
Non-X-ray деректерінде арнайы валидация жасалған жоқ. Бұл рентген екенін растайтын детектор емес.

СЫРТҚЫ OPENI, 200 СУРЕТ — ҚАЗІРГІ КОНФИГУРАЦИЯ
160 NORMAL, 40 PNEUMONIA, seed 42. Бұрынғы таңдау сақталды, шекті сыртқы деректе өзгертпедік.
CNN: шек 0.475, accuracy 21.50%, precision 20.30%, recall 100.00%, specificity 1.88%, F1 0.3376, ROC-AUC 0.4706; TN 3, FP 157, FN 0, TP 40.
MobileNetV2: шек 0.745, accuracy 55.00%, precision 23.96%, recall 57.50%, specificity 54.37%, F1 0.3382, ROC-AUC 0.5442; TN 87, FP 73, FN 17, TP 23.
Ensemble: шек 0.745, accuracy 55.00%, precision 23.96%, recall 57.50%, specificity 54.37%, F1 0.3382, ROC-AUC 0.5441; TN 87, FP 73, FN 17, TP 23.
SHA-256 ішкі train/validation/test қиылысы: 0. Бір report UID-інен бір сурет.
Report UID — surrogate топ, longitudinal patient ID толық расталмаған. MeSH gold standard емес.
Проекция PA деп кепілденбейді. Нәтиже тек осы таңдалған жиынға қатысты.

СТАТИСТИКА ЖӘНЕ ENSEMBLE
Exact paired McNemar: CNN-only 19, MobileNetV2-only 128,
p=4.8323820233e-21, alpha=0.05. Бұл ішкі test-ке қатысты.
Ensemble: validation-да таңдалған салмақтар {"CNN": 0.001005948939264576, "MobileNetV2": 0.9989940510607355}, шек 0.745. Сыртқы белгілер таңдау үшін қолданылған жоқ.

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
CI қашықта өтті (ubuntu-latest, Python 3.11, 56 тест). Run: https://github.com/5kairosh/Tynys/actions/runs/38745485393 . Тексерілген code commit: 53eeca479b9624e714d9e1a64173ad23761e8fff.
Репозиторий: https://github.com/5kairosh/Tynys
[![CI](https://github.com/5kairosh/Tynys/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/5kairosh/Tynys/actions/workflows/ci.yml)
docker build -t tynys .
docker run --rm -p 127.0.0.1:7860:7860 tynys
Docker: build/run тексерілді (image tynys, Server 29.8.1). Image көлемі: 3.28GB (3281444328 байт). Тексерілген host порт: 7861, контейнер: 7860. PNG/CR DICOM 200; CT DICOM/мәтінді JPG 400; PDF 200, 2 бет; Traceback жоқ. Контейнер жойылды.
Интернетке жарияланған жоқ.
Hugging Face Docker Space үшін түбірдегі README.md және Dockerfile дайын.

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


ҚҰЖАТТАР
presentation/tynys_complete_13slides_kz.pptx және tynys_project_speech_kz.txt — қазіргі қорғаныс нұсқасы.
MODEL_CARD.md — модель сипаттамасы, training/PHASE_REPORT_KZ.txt — кезеңдер есебі.
Training кэші мен OpenI суреттері таратылатын архивке кірмейді.

4-КЕЗЕҢ — САҚТЫҚ ЕРЕЖЕСІНІҢ НАҚТЫ ҚАМТУЫ МЕН ҚАТЕЛІГІ
Сайттың «Зерттеу нәтижесі» бөліміне ішкі test 618 / сыртқы OpenI 200 жеке
есептері және JSON жүктеу қосылды. Ішкі қамту 39/618 (6,31%), қабылданған топ
қатесі 6/39 (15,38%); OpenI 0/200, сондықтан таңдамалы accuracy есептелмейді.
Қателікті азайту расталған жоқ. Модель мен шектер өзгермеді.
Әдіс, шектеу, қайта орындау: training/SELECTIVE_EVALUATION_KZ.txt.

2026-10-20 — 4-кезеңді онлайн тексеру
Сайт: https://effective-space-winner-q74xxqj555vj39qq9-7860.app.github.dev/
GitHub CI сәтті: https://github.com/5kairosh/Tynys/actions/runs/39414009947
Тексерілген код: a4d1fe74662e32a7a89d2b2f2e5f76a0daa6d275
Жергілікті 109 pytest және 21 Chromium/IndexedDB тесті өтті; CI-де сол командалар
екі бөлек job ретінде сәтті аяқталды. Дәлел: training/selective_evidence/ci.json.
Ашық сайтта PNG және CR DICOM HTTP 200; CT DICOM және жалған JPG HTTP 400;
екі файл PDF есебі HTTP 200, 2 бет. JSON жүктеу /api/status есебімен дәл сәйкес.
Браузерде жаңа бөлім және нақты модель талдауы тексерілді; console error жоқ.
Codespace-та git жоқ: commit-тен 15 файл SHA-мен тексеріліп көшірілді, Git HEAD
жаңартылды деп есептелмейді. Қолданыстағы уақытша Codespaces шектеулері сақталады.
Презентацияның тарихи нұсқасы өзгертілмеді; 4-кезең нәтижелері осы құжат пен
training/SELECTIVE_EVALUATION_KZ.txt ішінде. Docker image бұл кезеңде қайта жиналған жоқ.
