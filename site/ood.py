"""Optional validation-distribution warning, not an X-ray authentication test."""
from pathlib import Path
import numpy as np


class OODDetector:
    def __init__(self):
        self.params = None
        self.message = 'OOD параметрлері жоқ. Validation белгілерінен ood_params.npz есептеу қажет.'

    def load(self, path, model_id):
        self.params = None
        try:
            with np.load(Path(path),allow_pickle=False) as data:
                mean = data['mean'].astype(np.float64)
                precision = data['precision'].astype(np.float64)
                threshold = float(data['threshold'])
                if str(data['model_id']) != model_id:
                    raise ValueError('Модель ID сәйкес емес.')
            if mean.ndim != 1 or precision.shape != (len(mean),len(mean)) or len(mean)!=1280:
                raise ValueError('Белгілер өлшемі сәйкес емес.')
            if not (np.isfinite(mean).all() and np.isfinite(precision).all() and np.isfinite(threshold) and threshold>0):
                raise ValueError('Параметрлер жарамсыз.')
            if not np.allclose(precision,precision.T,atol=1e-7):
                raise ValueError('Precision симметриялы болуы керек.')
            np.linalg.cholesky(precision)
            self.params = (mean,precision,threshold)
            self.message = None
        except Exception:
            self.message = 'Осы модельге жарамды OOD параметрлері жүктелмеді. ood_params.npz файлын тексеріңіз.'

    def status(self):
        return {'available':self.params is not None,'message':self.message,
                'notice':'Validation үлестірімінен ауытқу эвристикасы. Рентген екенін дәлелдемейді.'}

    def check(self, vector):
        if self.params is None:
            return self.status()
        mean,precision,threshold = self.params
        delta = np.asarray(vector,dtype=np.float64)-mean
        squared = float(delta @ precision @ delta)
        if not np.isfinite(squared) or squared < -1e-6:
            raise ValueError('Махаланобис қашықтығы жарамсыз.')
        distance = float(np.sqrt(max(0,squared)))
        return {'available':True,'is_xray':distance<=threshold,'distance':distance,'threshold':threshold,
                'notice':'is_xray — validation үлестіріміне ұқсастық эвристикасы, дәлелденген рентген детекторы емес.'}
