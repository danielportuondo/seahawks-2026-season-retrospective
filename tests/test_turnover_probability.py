import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase6_turnover_rate_model import zero_turnover_probability


def test_poisson_zero_matches_exp_formula():
    result = zero_turnover_probability(rate=0.05, exposure=20)
    assert result["p_zero_poisson"] == math.exp(-1.0)


def test_higher_rate_gives_lower_p_zero():
    low = zero_turnover_probability(rate=0.02, exposure=100)
    high = zero_turnover_probability(rate=0.08, exposure=100)
    assert high["p_zero_poisson"] < low["p_zero_poisson"]


def test_zero_exposure_gives_certainty():
    result = zero_turnover_probability(rate=0.5, exposure=0)
    assert result["p_zero_poisson"] == 1.0
    assert result["expected_turnovers"] == 0.0


def test_p_zero_bounded_zero_one():
    result = zero_turnover_probability(rate=0.0388, exposure=91.06)
    assert 0.0 <= result["p_zero_poisson"] <= 1.0
    assert result["odds_against_1_in"] == 1 / result["p_zero_poisson"]


def test_negative_binomial_omitted_without_alpha():
    result = zero_turnover_probability(rate=0.05, exposure=20)
    assert "p_zero_negative_binomial" not in result
    assert "odds_against_nb_1_in" not in result


def test_negative_binomial_conservative_bound():
    result = zero_turnover_probability(rate=0.0388, exposure=91.06, alpha=0.048)
    assert result["p_zero_negative_binomial"] > result["p_zero_poisson"]


def test_matches_project_reported_value():
    result = zero_turnover_probability(rate=20 / 516, exposure=91.05882352941177, alpha=0.047718364937986216)
    assert round(result["p_zero_poisson"], 4) == 0.0293
    assert round(result["p_zero_negative_binomial"], 4) == 0.0383
