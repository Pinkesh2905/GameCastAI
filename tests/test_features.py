"""
Feature engineering.

These are the tests that matter most in this project: the same functions run
at training time and at request time, so a regression here silently poisons
every prediction rather than raising anything.
"""

import pytest

from backend.ml.features import (
    MAX_RATE,
    ChaseState,
    SettingState,
    chase_features,
    phase_of,
    safe_rate,
    setting_features,
)


def chase(**overrides):
    base = dict(
        batting_team="A", bowling_team="B", target=181, score=95,
        balls_bowled=66, wickets_fallen=3, total_balls=120,
        venue="Somewhere", venue_par=165.0,
    )
    base.update(overrides)
    return chase_features(ChaseState(**base))


class TestSafeRate:
    def test_ordinary(self):
        assert safe_rate(90, 60) == 9.0

    def test_zero_balls_does_not_divide_by_zero(self):
        assert safe_rate(50, 0) == 0.0

    def test_absurd_asking_rate_is_capped(self):
        # 200 off one ball is not meaningfully different from impossible, and
        # an uncapped value would dominate the scaler.
        assert safe_rate(200, 1) == MAX_RATE


class TestPhase:
    @pytest.mark.parametrize("balls,expected", [
        (0, "powerplay"), (35, "powerplay"), (36, "middle"),
        (89, "middle"), (90, "death"), (119, "death"),
    ])
    def test_twenty_over_boundaries(self, balls, expected):
        assert phase_of(balls, 120) == expected

    def test_scales_to_a_rain_shortened_innings(self):
        # Ten overs in, a 60-ball game is at the death; a 120-ball game is not.
        assert phase_of(50, 60) == "death"
        assert phase_of(50, 120) == "middle"

    def test_zero_length_innings_does_not_crash(self):
        assert phase_of(0, 0) == "middle"


class TestChaseFeatures:
    def test_core_arithmetic(self):
        f = chase()
        assert f["runs_left"] == 86
        assert f["balls_left"] == 54
        assert f["wickets_left"] == 7
        assert f["current_run_rate"] == pytest.approx(95 * 6 / 66, rel=1e-3)
        assert f["required_run_rate"] == pytest.approx(86 * 6 / 54, rel=1e-3)

    def test_runs_left_never_negative(self):
        assert chase(score=200)["runs_left"] == 0

    def test_target_is_the_score_needed_to_win(self):
        # One short of the target still means the chase is not complete.
        assert chase(score=180)["runs_left"] == 1

    def test_wickets_left_clamps_at_both_ends(self):
        assert chase(wickets_fallen=12)["wickets_left"] == 0
        assert chase(wickets_fallen=0)["wickets_left"] == 10

    def test_balls_bowled_cannot_exceed_the_innings(self):
        f = chase(balls_bowled=999)
        assert f["balls_left"] == 0

    def test_target_vs_par_reads_off_the_ground(self):
        # The same target is a different proposition at a slow ground.
        assert chase(venue_par=150.0)["target_vs_par"] == 31.0
        assert chase(venue_par=190.0)["target_vs_par"] == -9.0

    def test_no_wickets_left_gives_a_finite_resource_number(self):
        f = chase(wickets_fallen=10)
        assert f["balls_per_wicket"] == 0.0
        assert f["runs_per_wicket_needed"] == 999.0

    def test_pressure_index_is_bounded(self):
        # An impossible chase should not send the feature to infinity.
        f = chase(target=400, score=0, balls_bowled=114)
        assert -10.0 <= f["pressure_index"] <= 20.0

    def test_team_identity_is_not_a_feature(self):
        # Removed deliberately: it made out-of-sample accuracy worse.
        f = chase()
        assert "batting_team" not in f
        assert "bowling_team" not in f
        assert "venue" not in f

    def test_shape_is_stable(self):
        from backend.ml.features import CHASE_FEATURES
        assert set(chase().keys()) == set(CHASE_FEATURES)


class TestSettingFeatures:
    def test_shape_is_stable(self):
        from backend.ml.features import SETTING_FEATURES
        f = setting_features(SettingState(
            batting_team="A", bowling_team="B", score=62,
            balls_bowled=36, wickets_fallen=1, venue_par=165.0,
        ))
        assert set(f.keys()) == set(SETTING_FEATURES)

    def test_start_of_innings(self):
        f = setting_features(SettingState(
            batting_team="A", bowling_team="B", score=0,
            balls_bowled=0, wickets_fallen=0,
        ))
        assert f["current_run_rate"] == 0.0
        assert f["balls_left"] == 120
        assert f["innings_progress"] == 0.0
