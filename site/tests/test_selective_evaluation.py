import csv
import hashlib
import json
from pathlib import Path

import pytest

from selective_evaluation import (artifact_bindings, binary_metrics,
                                  evaluation_status, summarize, validate_cohort)

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'site/model'


def test_selective_denominators_and_withheld_errors():
    result = summarize([
        {'label': 0, 'prediction': 0, 'xray_check': {'available': True, 'is_xray': True}},
        {'label': 1, 'prediction': 0, 'xray_check': {'available': True, 'is_xray': True}},
        {'label': 0, 'prediction': 1, 'xray_check': {'available': True, 'is_xray': False}},
        {'label': 1, 'prediction': 1, 'xray_check': {'available': False}},
    ])
    validate_cohort(result)
    assert result['coverage'] == .5
    assert result['baseline']['errors'] == 2
    assert result['accepted']['count'] == 2 and result['accepted']['errors'] == 1
    assert result['accepted']['error_rate'] == .5
    assert result['accepted']['sensitivity'] == 0.
    assert result['accepted']['specificity'] == 1.
    assert result['withheld']['errors'] == 1
    assert result['reasons'] == {'within_reference': 2, 'ood_shift': 1, 'ood_unavailable': 1}


@pytest.mark.parametrize('check', [None, {}, {'available': False},
                                  {'available': True, 'is_xray': 'true'},
                                  {'available': True, 'is_xray': False}])
def test_no_accepted_images_is_undefined_accuracy(check):
    result = summarize([{'label': 1, 'prediction': 1, 'xray_check': check}])
    validate_cohort(result)
    assert result['coverage'] == 0 and result['withheld_rate'] == 1
    assert result['accepted']['accuracy'] is None
    assert result['accepted']['error_rate'] is None
    assert result['accepted']['sensitivity'] is None
    assert result['accepted']['specificity'] is None


def test_absent_class_and_empty_cohort_are_not_zero_or_perfect():
    assert binary_metrics([0], [0])['sensitivity'] is None
    assert binary_metrics([1], [0])['specificity'] is None
    assert binary_metrics([], [])['accuracy'] is None
    assert summarize([])['coverage'] is None


@pytest.mark.parametrize('labels,predictions', [([1], []), ([2], [1]), ([True], [1]), ([1], [None])])
def test_invalid_binary_inputs_are_rejected(labels, predictions):
    with pytest.raises(ValueError):
        binary_metrics(labels, predictions)


def test_published_report_matches_each_per_image_record_and_old_baseline():
    report = json.loads((MODEL / 'selective_evaluation.json').read_text(encoding='utf-8'))
    assert report['thresholds_tuned_on_evaluation'] is False
    assert report['clinically_validated'] is False
    assert report['bindings'] == artifact_bindings(MODEL)
    for source, digest in report['source_sha256'].items():
        assert hashlib.sha256((ROOT / source).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest
    for cohort, count in [('internal_test', 618), ('external_openi', 200), ('rsna_confirmation', 400)]:
        with (ROOT / 'training/selective_evaluation' / (cohort + '.csv')).open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len({r['sha256'] for r in rows}) == count
        measured = summarize([{'label': int(r['label']), 'prediction': int(r['prediction']),
                              'xray_check': {'available': True, 'is_xray': float(r['distance']) <= float(r['ood_threshold'])}}
                             for r in rows])
        for key, value in measured.items():
            assert report['cohorts'][cohort][key] == value
        validate_cohort(report['cohorts'][cohort])
    internal = report['cohorts']['internal_test']['baseline']
    assert [internal[k] for k in ('TN', 'FP', 'FN', 'TP')] == [178, 53, 15, 372]
    external = report['cohorts']['external_openi']['baseline']
    assert [external[k] for k in ('TN', 'FP', 'FN', 'TP')] == [100, 60, 11, 29]


def test_current_report_is_available_from_real_status(client):
    status = client.get('/api/status').json()
    assert status['selective_evaluation']['available']
    assert status['selective_evaluation']['policy'] == status['assessment_policy']['name']
    assert status['ready'] and status['comparison_ready']


@pytest.mark.parametrize('bad', ['missing', 'json', 'binding', 'counts', 'null_accuracy', 'threshold_tuning'])
def test_invalid_or_stale_artifact_never_shows_as_current(tmp_path, monkeypatch, bad):
    report = json.loads((MODEL / 'selective_evaluation.json').read_text(encoding='utf-8'))
    current_bindings = report['bindings'].copy()
    monkeypatch.setattr('selective_evaluation.artifact_bindings', lambda model_dir: current_bindings)
    if bad == 'binding':
        report = {**report, 'bindings': {**report['bindings'], 'ood_parameters': 'changed'}}
    if bad == 'counts':
        report['cohorts']['internal_test']['accepted']['count'] += 1
    if bad == 'null_accuracy':
        report['cohorts']['internal_test']['accepted']['accuracy'] = None
    if bad == 'threshold_tuning':
        report['thresholds_tuned_on_evaluation'] = True
    if bad != 'missing':
        (tmp_path / 'selective_evaluation.json').write_text('broken' if bad == 'json' else json.dumps(report), encoding='utf-8')
    assert evaluation_status(tmp_path, report['bindings']['model'][:12], True)['available'] is False


def test_unavailable_ood_or_different_model_disables_only_report():
    assert evaluation_status(MODEL, 'other-model', True)['available'] is False
    assert evaluation_status(MODEL, artifact_bindings(MODEL)['model'][:12], False)['available'] is False
