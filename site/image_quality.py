"""Advisory pixel-quality checks, independent of disease inference and OOD.

Engineering thresholds; low detail is a reference-distribution percentile.
No claim of radiographic exposure,
anatomical coverage, X-ray authenticity, or diagnostic image quality is made.
The model receives its original TensorFlow-decoded pixels without alteration.
"""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

POLICY = 'image_quality_v1'
FEATURE_SCHEMA = 'gray_lanczos512_center90_laplacian4_over_robust_contrast_squared_v1'
REFERENCE_FILE = Path(__file__).resolve().parent/'model/image_quality_config.json'
THRESHOLDS = {
    'minimum_side_px': 224,
    'dark_p95': .18,
    'bright_p05': .82,
    'minimum_p95_p05_contrast': .10,
    'minimum_aspect_ratio': .45,
    'maximum_aspect_ratio': 2.2,
}
NOTICE = ('Бұл — пиксельдерге сүйенетін техникалық эвристика. '
          'Рентген екенін, клиникалық жарамдылығын немесе кеуденің толық көрінуін растамайды. '
          'Кеуденің қиылып қалмағанын түпнұсқадан өзіңіз тексеріңіз.')


def measure_quality(rgb):
    """Inspect decoded RGB pixels; never crop/normalize the model input.

Grayscale thumbnail (longest side <=512, no upscaling), then central 90%
limits border/label influence. Contrast is p95-p05 on [0,1]. Detail measure is
4-neighbor Laplacian variance divided by max(contrast**2,1e-6). Only flag low
detail when the original resolution and contrast are sufficient to measure it.
"""
    image = Image.fromarray(rgb).convert('L')
    width, height = image.size
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    gray = np.asarray(image, dtype=np.float64) / 255.
    border_y, border_x = int(gray.shape[0]*.05), int(gray.shape[1]*.05)
    roi = gray[border_y:gray.shape[0]-border_y, border_x:gray.shape[1]-border_x]
    p05, p95 = (float(n) for n in np.percentile(roi, [5, 95]))
    contrast = p95-p05
    detail = None
    if min(roi.shape) >= 3:
        laplacian = (roi[1:-1, :-2] + roi[1:-1, 2:] + roi[:-2, 1:-1]
                     + roi[2:, 1:-1] - 4*roi[1:-1, 1:-1])
        detail = float(np.var(laplacian) / max(contrast**2, 1e-6))
    return {'width': width, 'height': height, 'analysis_width': image.width,
            'analysis_height': image.height, 'p05': p05, 'p95': p95,
            'contrast': contrast, 'normalized_laplacian_variance': detail,
            'aspect_ratio': width/height}


def quality_reference():
    config = json.loads(REFERENCE_FILE.read_text(encoding='utf-8'))
    cutoff = config.get('detail_threshold')
    if (config.get('feature_schema') != FEATURE_SCHEMA or config.get('policy') != POLICY
            or type(cutoff) not in (int, float) or not math.isfinite(cutoff) or cutoff <= 0
            or type(config.get('eligible_count')) is not int or config['eligible_count'] < 100
            or config.get('reference_percentile') != 1.):
        raise ValueError('Image-quality reference configuration is invalid.')
    return config


def quality_status():
    try:
        reference = quality_reference()
        return {'available': True, 'policy': POLICY, 'clinically_validated': False,
                'reference_count': reference['reference_count'], 'detail_threshold': reference['detail_threshold']}
    except (OSError, ValueError, TypeError, KeyError):
        return {'available': False, 'policy': POLICY, 'clinically_validated': False,
                'message': 'Сапа тексеруінің анықтамалық параметрлері қолжетімсіз.'}


def check_image_quality(rgb):
    metrics = measure_quality(rgb)
    width, height = metrics['width'], metrics['height']
    p05, p95, contrast = metrics['p05'], metrics['p95'], metrics['contrast']
    detail = metrics['normalized_laplacian_variance']
    reference = quality_reference()
    warnings = []

    def warn(code, title, action):
        warnings.append({'code': code, 'title': title, 'action': action})

    if min(width, height) < THRESHOLDS['minimum_side_px']:
        warn('low_resolution', 'Сурет өлшемі шағын',
             'Қысқа жағы кемінде 224 пиксель болатын түпнұсқа файлды таңдаңыз; жай үлкейту жоғалған детальды қалпына келтірмейді.')
    if p95 < THRESHOLDS['dark_p95']:
        warn('very_dark', 'Суреттің көп бөлігі тым қараңғы',
             'Түпнұсқа файлды тексеріңіз. Экранды суретке түсіргеннен гөрі бастапқы файлды жүктеңіз.')
    if p05 > THRESHOLDS['bright_p05']:
        warn('very_bright', 'Суреттің көп бөлігі тым жарық',
             'Бастапқы файлды тексеріңіз; ақ аймақтарда ұсақ детальдар көрінбеуі мүмкін.')
    if contrast < THRESHOLDS['minimum_p95_p05_contrast']:
        warn('low_contrast', 'Суреттің контрасты төмен',
             'Ашық және қараңғы аймақтардың айырмасы аз. Бастапқы сападағы файлды қолданыңыз.')
    elif min(width, height) >= THRESHOLDS['minimum_side_px'] and detail is not None and detail < reference['detail_threshold']:
        warn('low_detail', 'Ұсақ деталь аз: сурет бұлыңғыр болуы мүмкін',
             'Фокусы анық түпнұсқаны жүктеңіз. Бұл белгі қозғалыс бұлыңғырлығын нақты растамайды.')
    ratio = width/height
    if not THRESHOLDS['minimum_aspect_ratio'] <= ratio <= THRESHOLDS['maximum_aspect_ratio']:
        warn('unusual_shape', 'Суреттің пішімі әдеттен тыс',
             'Кеуде толық көрінетінін және файлдың қатты қиылмағанын қолмен тексеріңіз.')
    return {
        'available': True, 'policy': POLICY, 'clinically_validated': False,
        'status': 'warning' if warnings else 'no_warning',
        'summary': 'Сурет сапасына назар аударыңыз' if warnings else 'Техникалық ескерту табылмады',
        'warnings': warnings,
        'metrics': metrics,
        'thresholds': {**THRESHOLDS, 'minimum_normalized_laplacian_variance': reference['detail_threshold']},
        'reference': {'count': reference['reference_count'], 'eligible_count': reference['eligible_count'],
                      'detail_percentile': reference['reference_percentile']}, 'notice': NOTICE,
        'affects_model_score': False, 'anatomical_crop_checked': False,
    }


def unavailable_quality():
    return {'available': False, 'policy': POLICY, 'status': 'unavailable',
            'summary': 'Сурет сапасын тексеру орындалмады', 'warnings': [],
            'notice': NOTICE, 'clinically_validated': False,
            'affects_model_score': False, 'anatomical_crop_checked': False}
