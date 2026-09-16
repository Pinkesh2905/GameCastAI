"""
Canonical feature engineering for GameCastAI.

Training and serving both call `chase_features()`. Keeping a single
implementation is deliberate: train/serve skew is the most common way a model
that looks good offline produces nonsense in production.

A note on what is deliberately absent. Batting and bowling team identity were
tested as model inputs and removed, because they measurably hurt out-of-sample
accuracy (holdout ROC-AUC fell from 0.903 to 0.895 on IPL). A T20 squad is
rebuilt every auction, so a name like "Mumbai Indians" is not a stable thing to
learn from a thousand matches, and the model was using it to memorise games.
The teams are still shown throughout the interface, and the venue survives as
`venue_par` because a ground genuinely is worth runs.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Chase (second innings) win-probability model ---------------------------

CHASE_NUMERIC = [
    "runs_left",
    "balls_left",
    "wickets_left",
    "current_run_rate",
    "required_run_rate",
    "pressure_index",
    "innings_progress",
    "target_rate",
    "target_vs_par",
    "runs_last_30",
    "wickets_last_30",
    "balls_per_wicket",
    "runs_per_wicket_needed",
]
CHASE_CATEGORICAL = ["phase"]
CHASE_FEATURES = CHASE_NUMERIC + CHASE_CATEGORICAL

# --- First-innings score projection model -----------------------------------

SETTING_NUMERIC = [
    "score",
    "balls_bowled",
    "balls_left",
    "wickets_left",
    "current_run_rate",
    "innings_progress",
    "venue_par",
    "runs_last_30",
    "wickets_last_30",
    "balls_per_wicket",
]
SETTING_CATEGORICAL = ["phase"]
SETTING_FEATURES = SETTING_NUMERIC + SETTING_CATEGORICAL

# A rate above this is arithmetically possible but never meaningfully
# different from "impossible", so we compress the tail rather than let a
# 200-run-off-6-balls situation dominate the scaler.
MAX_RATE = 36.0
UNKNOWN_VENUE = "Unknown Venue"
DEFAULT_PAR = 160.0


def phase_of(balls_bowled: int, total_balls: int) -> str:
    """Powerplay / middle / death, scaled to innings that rain has shortened."""
    if total_balls <= 0:
        return "middle"
    progress = balls_bowled / total_balls
    if progress < 0.30:
        return "powerplay"
    if progress < 0.75:
        return "middle"
    return "death"


def safe_rate(runs: float, balls: float) -> float:
    """Runs per over, guarding the zero-ball case and the runaway tail."""
    if balls <= 0:
        return 0.0
    return min(runs * 6.0 / balls, MAX_RATE)


@dataclass
class ChaseState:
    """Everything needed to describe a second innings mid-chase."""

    batting_team: str
    bowling_team: str
    target: int          # runs required to win
    score: int           # runs scored so far
    balls_bowled: int    # legal deliveries bowled so far
    wickets_fallen: int
    total_balls: int = 120
    venue: str = UNKNOWN_VENUE
    venue_par: float = DEFAULT_PAR
    runs_last_30: float = 0.0
    wickets_last_30: float = 0.0


def chase_features(state: ChaseState) -> dict:
    """Turn a chase situation into the exact feature dict the model expects."""
    total_balls = max(int(state.total_balls), 1)
    balls_bowled = min(max(int(state.balls_bowled), 0), total_balls)
    balls_left = total_balls - balls_bowled

    score = max(int(state.score), 0)
    # target is the score needed to WIN, so runs_left of 1 still means "not won".
    runs_left = max(int(state.target) - score, 0)
    wickets_left = min(max(10 - int(state.wickets_fallen), 0), 10)

    crr = safe_rate(score, balls_bowled)
    rrr = safe_rate(runs_left, balls_left) if balls_left > 0 else MAX_RATE
    venue_par = float(state.venue_par or DEFAULT_PAR)

    return {
        "runs_left": float(runs_left),
        "balls_left": float(balls_left),
        "wickets_left": float(wickets_left),
        "current_run_rate": round(crr, 4),
        "required_run_rate": round(rrr, 4),
        # How much harder the chase has to get than it has been so far.
        "pressure_index": round(max(min(rrr - crr, 20.0), -10.0), 4),
        "innings_progress": round(balls_bowled / total_balls, 4),
        # Par difficulty of the chase as a whole, which lets one model serve
        # a 120-ball chase and a rain-reduced 60-ball one.
        "target_rate": round(safe_rate(state.target, total_balls), 4),
        # 180 is a stiff chase at a slow ground and an ordinary one at a fast
        # ground; this is what carries that difference.
        "target_vs_par": round(float(state.target) - venue_par, 3),
        "runs_last_30": float(state.runs_last_30),
        "wickets_last_30": float(state.wickets_last_30),
        # Batting resources: deliveries and runs available per wicket in hand.
        "balls_per_wicket": round(balls_left / wickets_left, 4) if wickets_left else 0.0,
        "runs_per_wicket_needed": round(runs_left / wickets_left, 4) if wickets_left else 999.0,
        "phase": phase_of(balls_bowled, total_balls),
    }


@dataclass
class SettingState:
    """First innings, where the question is what total the side will post."""

    batting_team: str
    bowling_team: str
    score: int
    balls_bowled: int
    wickets_fallen: int
    total_balls: int = 120
    venue: str = UNKNOWN_VENUE
    venue_par: float = DEFAULT_PAR
    runs_last_30: float = 0.0
    wickets_last_30: float = 0.0


def setting_features(state: SettingState) -> dict:
    total_balls = max(int(state.total_balls), 1)
    balls_bowled = min(max(int(state.balls_bowled), 0), total_balls)
    balls_left = total_balls - balls_bowled
    score = max(int(state.score), 0)
    wickets_left = min(max(10 - int(state.wickets_fallen), 0), 10)

    return {
        "score": float(score),
        "balls_bowled": float(balls_bowled),
        "balls_left": float(balls_left),
        "wickets_left": float(wickets_left),
        "current_run_rate": round(safe_rate(score, balls_bowled), 4),
        "innings_progress": round(balls_bowled / total_balls, 4),
        "venue_par": float(state.venue_par or DEFAULT_PAR),
        "runs_last_30": float(state.runs_last_30),
        "wickets_last_30": float(state.wickets_last_30),
        "balls_per_wicket": round(balls_left / wickets_left, 4) if wickets_left else 0.0,
        "phase": phase_of(balls_bowled, total_balls),
    }
