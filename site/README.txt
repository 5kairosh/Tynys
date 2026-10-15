TYNYS — КЕҢЕЙТІЛГЕН ОҚУ ЖОБАСЫ

ІСКЕ ҚОСУ
Windows: Python 3.11 орнатып, архивті толық шығарып, site/start.bat бас.
Браузер: http://127.0.0.1:7860 . Сервер терезесін ашық қалдыр.
Жаңа виртуалды ортада requirements автоматты орнатылады. Бұрынғы ортада:
site/.venv/Scripts/python -m pip install -r site/requirements.txt
Сервер үзілсе қосылым әр 3 секунд сайын қайта тексеріледі.

НЕ ЖҰМЫС ІСТЕЙДІ
JPG/PNG және CR/DX DICOM, әр файл 12 МБ-қа дейін, бір топта 10 суретке дейін.
Екі модельдің санаты, баллы, өз шешім шегі, салыстыруы және Grad-CAM.
Түпнұсқа/жылу картасын ауыстыру, PNG, мәтін және PDF жүктеу.
Осы бет жадындағы соңғы 20 талдау, қайта ашу және тазалау.
Тарих localStorage/IndexedDB/cookie-ге жазылмайды, бетті жаңартқанда жойылады.
Қабылданған multipart жүктеулер жадта өңделеді. Суреттер сервер файлына жазылмайды.
Тек зерттеуге арналған оқу прототипі. Медициналық диагноз емес.
NORMAL жауабы барлық ауруды жоққа шығармайды.

ИНФЕРЕНС
JPG/PNG: tf.io.decode_image(channels=3,expand_animations=False), float32,
resize_with_pad(224,224), модель. EXIF rotation, сыртқы нормализация және crop жоқ.
Rescaling модель ішінде. DICOM: CR/DX, бір MONOCHROME кадр, Modality LUT,
1–99 процентиль, MONOCHROME1 инверсиясы, жадта RGB PNG, кейін сол TF pipeline.
CT/MR, зақымдалған немесе бос DICOM қабылданбайды. Сығылған transfer syntax
үшін бөлек кодек қажет болуы мүмкін. pydicom жоқ болса функция ғана қолжетімсіз.

НАҚТЫ ІШКІ НӘТИЖЕ (КАЛИБРЛЕУСІЗ)
Test 618, train 3801, validation 952. MobileNetV2 шегі 0.740.
Accuracy 89.16%, precision 87.74%, recall 96.12%, specificity 77.49%,
F1 0.9174, ROC-AUC 0.9581. TN 179, FP 52, FN 15, TP 372.
CNN accuracy 71.68%, өз шегі 0.485. 90% accuracy мақсатына жетпеді.
McNemar exact two-sided: CNN-only correct 20, MobileNetV2-only correct 128,
p=1.8041385810152942e-20. Бұл осы ішкі тестке қатысты.

СЫРТҚЫ ТЕКСЕРУ (ОРЫНДАЛДЫ)
NLM OpenI, 200 сурет: 160 NORMAL, 40 PNEUMONIA, seed 42.
Шек өзгермеген (0.740). Accuracy 55.00%, precision 23.96%, recall 57.50%,
specificity 54.38%, F1 0.3382, ROC-AUC 0.5442. TN 87, FP 73, FN 17, TP 23.
Ішкі train/validation/test SHA-256-мен қиылысу: 0. Бір есептен бір сурет.
Есеп UID-і — топтық surrogate ID. Пациенттің ұзақ мерзімді ID-і толық расталмаған.
MeSH белгілері gold standard емес. Проекцияны PA деп кепілдемейміз.
Сыртқы төмен нәтиже ішкі accuracy-ді кез келген рентгенге таратуға болмайтынын көрсетеді.
Есептер: training/external_validation_report.txt, external_validation_metrics.csv,
external_predictions.csv, external_confusion_matrix.csv.

КАЛИБРЛЕУ ЖӘНЕ OOD (КОДЫ ДАЙЫН, ПАРАМЕТРЛЕРІ ЕСЕПТЕЛМЕГЕН)
Colab-қа temperature scaling ұяшығы қосылды. T validation NLL бойынша 0.2–5.0
арасында таңдалады. Шек сол validation-да recall>=0.90, specificity, F1 бойынша қайта таңдалады.
Test тек кейін бағаланады. Әр модель өз T мәнін қолданады.
Grad-CAM reconstructed_score raw sigmoid-пен тексеріледі, калибрленген баллмен емес.
Калибрлеу клиникалық ықтималдық ретінде тексерілгенін білдірмейді.
Қазір model_config.json-да calibration жоқ: T=1, бастапқы болжам дәл сақталады.
Validation-ның 312/952 түпнұсқа файлы SHA-256 арқылы тексерілді, одан кейін
storage.googleapis.com TLS/read timeout үш рет қайталанды. Жартылай validation-мен T таңдамадық.
OOD коды validation-ның 1280 белгілерінен shrinkage covariance, mean және 95-процентиль
Махаланобис шегін есептейді. ood_params.npz жоқ болғандықтан сайт «қолжетімсіз» дейді.
OOD параметрлері қосылса болжамды блоктамайды, тек ескерту береді, PDF-ке де жазады.
Бұл non-X-ray деректерінде валидацияланған рентген детекторы емес.

ҚАЙТА ЕСЕПТЕУ (ЖОБА ТҮБІРІНЕН)
site/.venv/Scripts/python -m pip install -r training/requirements-training.txt
site/.venv/Scripts/python training/download_validation.py
site/.venv/Scripts/python training/prepare_validation_artifacts.py
site/.venv/Scripts/python training/external_validation.py --openi --count 200
Соңынан серверді қайта іске қос. Алғашқы скрипт кэштен жарамды SHA-256 файлдарын қайта қолданады.
Немесе training/pneumonia_colab_full_kz.ipynb ұяшықтарын Colab-та ретімен орында.
Жаңа архивтен модель, конфигурация, метрикалар, mcnemar.json, ood_params.npz бірге көшіріледі.
Калибрлеуден кейін ескі external_validation.json конфигурация хэші сәйкес болмаса қолданылмайды.
Датасет кэші тек training/cache ішінде, бұл пайдаланушының сайтқа жүктеген суреттері емес.
Кэш пен сыртқы суреттер таратылатын ZIP-ке кірмейді.

ТЕСТТЕР ЖӘНЕ CI
site/.venv/Scripts/python -m pip install pytest pypdf httpx
site/.venv/Scripts/python -m pytest site/tests -q
34 тест өтті: config, calibration T=1 сәйкестігі, pad геометриясы, TF padding,
PDF беттері, API, DICOM CR/CT, бос/бүлінген/13 МБ файлдар, OOD fallback және топтық таңдау.
.github/workflows/ci.yml Ubuntu/Python 3.11 үшін дайын. Қашықтағы CI іске қосылған жоқ.

DOCKER / HUGGING FACE
Түбірдегі Dockerfile: docker build -t tynys .
Іске қосу: docker run --rm -p 127.0.0.1:7860:7860 tynys
site/Dockerfile үшін: docker build -t tynys -f site/Dockerfile site
README.md Hugging Face Docker Space frontmatter қамтиды, app_port 7860.
Docker Desktop Linux daemon қосылмаған: build/run тексерілген жоқ. Интернетке жарияланған жоқ.
Модель картасы: MODEL_CARD.md. Лицензия шарттарын жариялау алдында тексеру қажет.

ПРЕЗЕНТАЦИЯ
Соңғы нұсқа: presentation/tynys_project_12slides_kz.pptx.
Айтылым: presentation/tynys_project_speech_kz.txt.
Сыртқы валидация мен калибрлеудің нақты орындалу күйіне 2 слайд қосылды.
Multi-label, ensemble және жаңа аугментациялар енгізілген жоқ: 3-кезең шарты толық орындалмаған.
