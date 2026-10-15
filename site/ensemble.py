"""Optional validation-fitted convex ensemble; never modifies base predictions."""
import math


def validate_ensemble(value):
    if not isinstance(value, dict):
        raise ValueError('Ensemble конфигурациясы жоқ.')
    weights = value.get('weights')
    if not isinstance(weights, dict) or set(weights) != {'CNN', 'MobileNetV2'}:
        raise ValueError('Екі модельдің салмағы қажет.')
    values = [*weights.values(), value.get('threshold')]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Ensemble сандары жарамсыз.')
    if any(not 0 < v < 1 for v in weights.values()) or abs(sum(weights.values())-1) > 1e-9:
        raise ValueError('Салмақтар оң, қосындысы 1 болуы керек.')
    if not 0 <= value['threshold'] <= 1:
        raise ValueError('Ensemble шегі жарамсыз.')
    return value


def ensemble_score(cnn, mobile, config):
    validate_ensemble(config)
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in (cnn, mobile)):
        raise ValueError('Модель баллы жарамсыз.')
    return config['weights']['CNN']*cnn + config['weights']['MobileNetV2']*mobile
