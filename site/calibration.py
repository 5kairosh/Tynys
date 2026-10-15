"""Temperature scaling after the unchanged model pipeline."""
import math


def validate_calibration(value):
    if value is None:
        return
    if not isinstance(value,dict) or value.get('method') != 'temperature':
        raise ValueError('Калибрлеу әдісі temperature болуы керек.')
    temperature = value.get('T')
    if isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or not math.isfinite(temperature) or not .2 <= temperature <= 5:
        raise ValueError('Калибрлеу T мәні 0,2–5,0 арасында болуы керек.')


def calibrate_score(score, calibration=None):
    validate_calibration(calibration)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Модель баллы жарамсыз.')
    temperature = calibration['T'] if calibration else 1.
    if temperature == 1:
        return float(score)  # Exact legacy parity, including 0 and 1.
    p = min(max(score,1e-7),1-1e-7)
    z = (math.log(p)-math.log1p(-p))/temperature
    return 1/(1+math.exp(-z))
