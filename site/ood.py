"""Train reference + separate validation cutoff; not an X-ray authenticator."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


class OODDetector:
    def __init__(self):
        self.params = None
        self.metadata = None
        self.parameter_id = None
        self.message = 'OOD параметрлері жоқ. Train/validation бойынша calibrate_ood.py орындау қажет.'

    def load(self, path, model_id, config_sha256=None):
        self.params = None
        self.metadata = None
        self.parameter_id = None
        try:
            with np.load(Path(path),allow_pickle=False) as data:
                mean = data['mean'].astype(np.float64)
                precision = data['precision'].astype(np.float64)
                threshold = float(data['threshold'])
                metadata = json.loads(str(data['metadata']))
                if str(data['model_id']) != model_id:
                    raise ValueError('Модель ID сәйкес емес.')
            if (metadata['schema'] != 2 or metadata['method'] != 'train_validation_mahalanobis_v2'
                    or metadata['reference_split'] != 'train' or metadata['calibration_split'] != 'validation'
                    or type(metadata['reference_count']) is not int or metadata['reference_count'] < 2
                    or type(metadata['calibration_count']) is not int or metadata['calibration_count'] < 20
                    or metadata['target_acceptance'] != .95
                    or metadata['quantile_rank'] != math.ceil((metadata['calibration_count']+1)*.95)
                    or metadata['test_or_external_used_to_fit'] is not False
                    or metadata['clinical_or_formal_coverage_guarantee'] is not False
                    or metadata['model_sha256'][:12] != model_id
                    or (config_sha256 is not None and metadata['config_sha256'] != config_sha256)):
                raise ValueError('Train/validation provenance жарамсыз.')
            if mean.ndim != 1 or precision.shape != (len(mean),len(mean)) or len(mean)!=1280:
                raise ValueError('Белгілер өлшемі сәйкес емес.')
            if not (np.isfinite(mean).all() and np.isfinite(precision).all() and np.isfinite(threshold) and threshold>0):
                raise ValueError('Параметрлер жарамсыз.')
            if not np.allclose(precision,precision.T,atol=1e-7):
                raise ValueError('Precision симметриялы болуы керек.')
            np.linalg.cholesky(precision)
            self.params = (mean,precision,threshold)
            self.metadata = metadata
            self.parameter_id = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
            self.message = None
        except Exception:
            self.message = 'Осы модельге жарамды OOD параметрлері жүктелмеді. ood_params.npz файлын тексеріңіз.'

    def status(self):
        return {'available':self.params is not None,'message':self.message,
                'parameter_id':self.parameter_id, 'method':(self.metadata or {}).get('method'),
                'reference_count':(self.metadata or {}).get('reference_count'),
                'calibration_count':(self.metadata or {}).get('calibration_count'),
                'target_acceptance':(self.metadata or {}).get('target_acceptance'),
                'threshold':self.params[2] if self.params is not None else None,
                'notice':'Train үлестіріміне ұқсастық; шек бөлек validation-да бапталған. Рентген екенін немесе болжамның дұрыстығын дәлелдемейді.'}

    def check(self, vector):
        if self.params is None:
            return self.status()
        mean,precision,threshold = self.params
        value = np.asarray(vector,dtype=np.float64)
        if value.shape != mean.shape or not np.isfinite(value).all():
            raise ValueError('OOD белгі векторының өлшемі не мәндері жарамсыз.')
        delta = value-mean
        squared = float(delta @ precision @ delta)
        if not np.isfinite(squared) or squared < -1e-6:
            raise ValueError('Махаланобис қашықтығы жарамсыз.')
        distance = float(np.sqrt(max(0,squared)))
        return {'available':True,'is_xray':distance<=threshold,'within_reference':distance<=threshold,
                'distance':distance,'threshold':threshold,'parameter_id':self.parameter_id,
                'method':(self.metadata or {}).get('method'),
                'notice':'within_reference (is_xray — ескі API атауы) — үлестірімге ұқсастық. Нақты рентген де басқа дереккөзден болса ауытқуы мүмкін; бұл X-ray детекторы емес.'}
