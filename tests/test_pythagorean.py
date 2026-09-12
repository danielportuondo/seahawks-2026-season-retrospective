import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pythagorean import expected_wins, pythagorean_win_pct


def test_equal_points_gives_half():
    assert pythagorean_win_pct(300, 300) == 0.5


def test_more_points_for_gives_majority():
    assert pythagorean_win_pct(400, 300) > 0.5


def test_more_points_against_gives_minority():
    assert pythagorean_win_pct(300, 400) < 0.5


def test_win_pct_bounded_zero_one():
    assert 0.0 <= pythagorean_win_pct(1, 1000) <= 1.0
    assert 0.0 <= pythagorean_win_pct(1000, 1) <= 1.0


def test_expected_wins_scales_with_games():
    assert expected_wins(400, 300, 17) == pythagorean_win_pct(400, 300) * 17
