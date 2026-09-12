"""Pythagorean win expectation — pure functions, unit tested.

Uses the NFL-calibrated exponent (~2.37, Football Outsiders) rather than
the classic baseball exponent of 2.
"""

NFL_EXPONENT = 2.37


def pythagorean_win_pct(points_for: float, points_against: float, exponent: float = NFL_EXPONENT) -> float:
    if points_for <= 0 and points_against <= 0:
        return 0.5
    pf_exp = points_for**exponent
    pa_exp = points_against**exponent
    return pf_exp / (pf_exp + pa_exp)


def expected_wins(points_for: float, points_against: float, games: int, exponent: float = NFL_EXPONENT) -> float:
    return pythagorean_win_pct(points_for, points_against, exponent) * games
