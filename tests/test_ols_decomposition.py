import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase7_decomposition import fit_ols


def _synthetic_data(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    true_coefs = np.array([3.0, -1.5])
    noise = rng.normal(scale=0.05, size=n)
    y = 2.0 + X @ true_coefs + noise
    return X, y, true_coefs


def test_recovers_known_coefficients():
    X, y, true_coefs = _synthetic_data()
    result = fit_ols(X, y, ["a", "b"])
    assert abs(result["coefficients"]["a"] - true_coefs[0]) < 0.1
    assert abs(result["coefficients"]["b"] - true_coefs[1]) < 0.1
    assert abs(result["coefficients"]["intercept"] - 2.0) < 0.1


def test_r2_near_one_for_low_noise_fit():
    X, y, _ = _synthetic_data()
    result = fit_ols(X, y, ["a", "b"])
    assert 0.95 <= result["r2"] <= 1.0
    assert result["r2_adj"] <= result["r2"]


def test_p_values_are_valid_probabilities():
    X, y, _ = _synthetic_data()
    result = fit_ols(X, y, ["a", "b"])
    for p in result["p_values"].values():
        assert 0.0 <= p <= 1.0


def test_dof_matches_n_minus_predictors_minus_intercept():
    X, y, _ = _synthetic_data(n=64)
    result = fit_ols(X, y, ["a", "b"])
    assert result["dof_resid"] == 64 - 2 - 1
