MULTI-LABEL: ЖЕКЕ ОҚУ ЭКСПЕРИМЕНТІ
617 сурет, 331 report тобы. Train434 / validation91 / test92.
51 суретте кемінде екі оң MeSH белгісі бар. SHA-256 және report/group қиылысы: 0.
Белгі: pneumonia, atelectasis (MeSH: Pulmonary Atelectasis), cardiomegaly.
ImageNet MobileNetV2 + GAP + Dropout + үш бөлек sigmoid басы, BCE.
Head3 epoch + соңғы30 қабат2 epoch, BN мұздатылған, таңдалған кезең: fine.
Әр шек validation F1 арқылы таңдалды, test шек таңдауға қатыспады.

pneumonia: шек 0.26, AUC 0.5378, F1 0.2308, sensitivity 30.00%, specificity 84.15%; TN69 FP13 FN7 TP3.
atelectasis: шек 0.17, AUC 0.6196, F1 0.4828, sensitivity 96.55%, specificity 6.35%; TN4 FP59 FN1 TP28.
cardiomegaly: шек 0.18, AUC 0.5632, F1 0.4894, sensitivity 79.31%, specificity 33.33%; TN21 FP42 FN6 TP23.

Бұл шағын квоталық оқу эксперименті. Сапасы клиникалық қолдануға жеткіліксіз.
MeSH report деңгейінде, бір суреттегі нақты ауруды растамайды. Жоқ белгі weak negative деп алынды.
Report UID нақты longitudinal patient ID емес. Frontal/lateral проекциялар аралас.
Binary сыртқы200 cohort-пен71 сурет қиылысады; бұл жаңа тәуелсіз binary тексеру емес.

ЖЕРГІЛІКТІ ҒАНА
private/openi_multilabel.keras — осы деректе үйретілген модель.
private/head.weights.h5 — validation арқылы кезеңді салыстыру үшін checkpoint.
private/error_01.png ... error_08.png — сегіз қате мысалы.
OpenI суреттері training/cache/openi_multilabel ішінде.
Тапсырма талабына сәйкес бұл суреттерді, қате суреттерін және салмақтарды таратуға болмайды.
Олар Git пен release ZIP-ке кірмейді. Архив тек мәтіндік дәлелдер мен суретсіз графиктерді қамтиды.

ҚАЙТА ОРЫНДАУ
Жоба түбірінен:
site/.venv/Scripts/python -m pip install -r training/requirements-training.txt
site/.venv/Scripts/python training/multilabel_experiment.py
Бар манифестпен және SHA-256 тексеруімен:
site/.venv/Scripts/python training/multilabel_experiment.py --reuse-manifests
Colab үшін: training/multilabel_colab_kz.ipynb (өздігінен орындалатын қазақша ноутбук).
Ноутбук Colab-та осы сессияда орындалған жоқ; дәл сол функциялар жергілікті TensorFlow2.21-де орындалды.
Сайтқа бұл модель қосылған жоқ. NORMAL/PNEUMONIA сайт модельдері сақталған.
Дереккөз: https://openi.nlm.nih.gov/ , лицензия сілтемесі data_report.json ішінде.
