# Data sources for the Stage 6 educational experiment

The **NIH Clinical Center is the provider** of the original NIH chest radiographs.
Original NIH download: https://nihcc.app.box.com/v/ChestXray-NIHCC

Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers.
*ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised
Classification and Localization of Common Thorax Diseases*. IEEE CVPR, pp. 3462–3471, 2017.

The **RSNA and Society of Thoracic Radiology (STR)** supplied the adjudicated
challenge radiographic annotations and source mappings.
[Official download and terms](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).
Academic/educational use is covered by these source terms. Do not attempt to identify or contact subjects.

George Shih, Carol C. Wu, Safwan S. Halabi, Marc D. Kohli, Luciano M. Prevedello,
Tessa S. Cook, Arjun Sharma, Judith K. Amorosa, Veronica Arteaga, Maya Galperin-Aizenberg,
Ritu R. Gill, Myrna C. B. Godoy, Stephen Hobbs, Jean Jeudy, Archana Laroia, Palmi N. Shah,
Dharshan Vummidi, Kavitha Yaddanapudi, Anouk Stein.
*Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert
Annotations of Possible Pneumonia*. Radiology: Artificial Intelligence, 2019.
https://doi.org/10.1148/ryai.2019180041

Stage 6 uses the **Calculated Normal vs Lung Opacity** labels; other abnormalities
are excluded, never silently mapped to NORMAL. Lung opacity is a radiographic
proxy, not clinically confirmed pneumonia. The quota sample is balanced by design.
The final confirmation is from new NIH patient groups within the same RSNA source
used for development; it is not validation at a new institution.

The prior Kaggle dataset and its existing attribution remain part of the project:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

OpenI Indiana is **evaluation-only**, with weak report-level MeSH labels:
https://openi.nlm.nih.gov/ . Its metadata links CC BY-NC-ND 4.0.
No OpenI image pixels, error overlays, or OpenI-trained weights are redistributed.
No user-uploaded images are part of the experiment, repository or release ZIP.
All source image/DICOM/feature caches are ignored and excluded from releases.
