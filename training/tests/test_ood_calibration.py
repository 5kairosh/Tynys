import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('calibrate_ood', Path(__file__).resolve().parents[1]/'calibrate_ood.py')
fit = importlib.util.module_from_spec(spec); spec.loader.exec_module(fit)


def test_reference_geometry_does_not_depend_on_calibration_values():
    rng = np.random.default_rng(67)
    train = rng.normal(size=(60, 40))
    calibration = rng.normal(size=(120, 40))
    holdout = rng.normal(size=(400, 40))
    mean, precision = fit.fit_reference(train)
    assert np.allclose(mean, train.mean(axis=0))
    assert np.linalg.eigvalsh(precision).min() > 0
    threshold, rank = fit.calibrate_cutoff(fit.distances(calibration, mean, precision))
    old_threshold = np.quantile(fit.distances(train, mean, precision), .95)
    d = fit.distances(holdout, mean, precision)
    assert threshold > old_threshold
    assert np.mean(d <= threshold) > .8  # independent same-distribution controls
    assert np.mean(d <= threshold) > np.mean(d <= old_threshold) + .2
    assert rank == 115
    assert np.allclose(mean, train.mean(axis=0))


def test_prespecified_order_statistic_including_ties():
    threshold, rank = fit.calibrate_cutoff(np.arange(1, 953, dtype=float))
    assert rank == 906 and threshold == 906
    assert fit.calibrate_cutoff(np.ones(100))[0] == 1.


@pytest.mark.parametrize('values', [[], [float('nan')]*30, [-1]*30, [0]*30, [1]*10, [[1]*30]])
def test_invalid_calibration_is_not_published(values):
    with pytest.raises(ValueError):
        fit.calibrate_cutoff(values)


@pytest.mark.parametrize('values', [np.ones((1,3)), np.full((3,3),np.inf), np.ones(3)])
def test_invalid_reference_matrix_is_rejected(values):
    with pytest.raises(ValueError):
        fit.fit_reference(values)
