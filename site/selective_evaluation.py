"""Audit a fixed abstention policy; never fit a threshold or change predictions."""
import hashlib
import json
from pathlib import Path

from assessment import POLICY, assess_input

SCHEMA = 1


def binary_metrics(labels, predictions):
    if len(labels) != len(predictions):
        raise ValueError('Labels and predictions must have equal length.')
    if any(type(v) is not int or v not in (0, 1) for v in [*labels, *predictions]):
        raise ValueError('Binary integer labels/predictions are required.')
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    n = len(labels)
    return {'count': n, 'normal': tn + fp, 'pneumonia': fn + tp,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'errors': fp + fn,
            'accuracy': (tn + tp) / n if n else None,
            'error_rate': (fp + fn) / n if n else None,
            'sensitivity': tp / (tp + fn) if tp + fn else None,
            'specificity': tn / (tn + fp) if tn + fp else None}


def summarize(rows):
    """Rows contain real predictions plus the check passed to assess_input()."""
    groups = {'baseline': [], 'accepted': [], 'withheld': []}
    reasons = {'within_reference': 0, 'ood_shift': 0, 'ood_unavailable': 0}
    for row in rows:
        decision = assess_input(row.get('xray_check'))
        reasons[decision['reason']] += 1
        groups['baseline'].append(row)
        groups['accepted' if decision['status'] == 'classified' else 'withheld'].append(row)
    result = {name: binary_metrics([r['label'] for r in values],
                                   [r['prediction'] for r in values])
              for name, values in groups.items()}
    n = result['baseline']['count']
    result.update(coverage=result['accepted']['count'] / n if n else None,
                  withheld_rate=result['withheld']['count'] / n if n else None,
                  reasons=reasons)
    return result


def artifact_bindings(model_dir):
    root = Path(__file__).resolve().parent
    sources = {'model': Path(model_dir) / 'pneumonia_mobilenetv2.keras',
               'config': Path(model_dir) / 'model_config.json',
               'ood_parameters': Path(model_dir) / 'ood_params.npz',
               'assessment': root / 'assessment.py', 'ood_code': root / 'ood.py',
               'calibration_code': root / 'calibration.py', 'api_pipeline': root / 'app.py',
               'metrics_code': root / 'selective_evaluation.py'}
    # Git text files may be checked out as CRLF on Windows. Bind their logical
    # source bytes, while preserving EXACT model/config/NPZ bytes.
    return {name: hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')
                                if path.suffix == '.py' else path.read_bytes()).hexdigest()
            for name, path in sources.items()}


def validate_cohort(value):
    for key in ('baseline', 'accepted', 'withheld'):
        stats = value[key]
        counts = [stats[k] for k in ('TN', 'FP', 'FN', 'TP')]
        if any(type(n) is not int or not 0 <= n <= 100000 for n in counts) or sum(counts) > 100000:
            raise ValueError('Invalid confusion counts.')
        tn, fp, fn, tp = counts
        # Reconstruct numerical denominators, not patient records.
        expected = binary_metrics([0] * (tn + fp) + [1] * (fn + tp),
                                  [0] * tn + [1] * fp + [0] * fn + [1] * tp)
        if stats != expected:
            raise ValueError('Metrics do not match confusion counts/denominators.')
    total, accepted, withheld = (value[k] for k in ('baseline', 'accepted', 'withheld'))
    for key in ('count', 'normal', 'pneumonia', 'TN', 'FP', 'FN', 'TP', 'errors'):
        if total[key] != accepted[key] + withheld[key]:
            raise ValueError('Accepted/withheld partitions are inconsistent.')
    if total['count'] <= 0:
        raise ValueError('A published cohort must not be empty.')
    if value['coverage'] != accepted['count'] / total['count'] or value['withheld_rate'] != withheld['count'] / total['count']:
        raise ValueError('Invalid coverage denominator.')
    reasons = value['reasons']
    if set(reasons) != {'within_reference', 'ood_shift', 'ood_unavailable'} or any(type(v) is not int or v < 0 for v in reasons.values()):
        raise ValueError('Invalid reasons.')
    if reasons['within_reference'] != accepted['count'] or reasons['ood_shift'] + reasons['ood_unavailable'] != withheld['count']:
        raise ValueError('Reason counts are inconsistent.')


def evaluation_status(model_dir, model_id, ood_available):
    unavailable = {'available': False, 'message': 'Осы модель мен OOD ережесіне сәйкес қамту/қателік есебі қолжетімсіз.'}
    try:
        value = json.loads((Path(model_dir) / 'selective_evaluation.json').read_text(encoding='utf-8'))
        if not ood_available or value['schema'] != SCHEMA or value['policy'] != POLICY or value['clinically_validated'] is not False:
            raise ValueError('Unavailable policy or incompatible schema.')
        if value['bindings'] != artifact_bindings(model_dir) or value['bindings']['model'][:12] != model_id:
            raise ValueError('Evaluation belongs to different source/model/parameters.')
        if value['thresholds_tuned_on_evaluation'] is not False:
            raise ValueError('Holdout must not tune thresholds.')
        for key in ('internal_test', 'external_openi'):
            validate_cohort(value['cohorts'][key])
        return {'available': True, **value}
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return unavailable
