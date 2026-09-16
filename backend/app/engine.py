"""
The prediction engine.

Loads one model bundle per league and answers the questions the interface
asks: what is the win probability now, what would it be after a given over,
how does it move across the range of scores, and what did it look like ball by
ball in a real match.

Everything here is pure and side-effect free apart from the lazy model load,
which keeps the API layer thin and makes the logic straightforward to test.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from backend.app.config import PROJECT_ROOT
from backend.app.inference import ChaseModel, SettingModel
from backend.ml.features import (
    ChaseState,
    SettingState,
    chase_features,
    phase_of,
    setting_features,
)
from backend.ml.leagues import BY_CODE, DEFAULT_LEAGUE
from backend.ml.priors import Priors

MODEL_DIR = PROJECT_ROOT / "models"

# Five overs, matching the rolling window the models were trained on.
FORM_WINDOW = 30


class LeagueNotAvailable(Exception):
    """Raised when a league has no trained model on disk."""


@dataclass
class Bundle:
    """Everything needed to serve one league."""

    code: str
    meta: dict
    chase_model: ChaseModel
    setting_model: SettingModel | None
    priors: Priors


@lru_cache(maxsize=None)
def load_bundle(code: str) -> Bundle:
    """Load a league from the exported artefacts, not the training pickles.

    `backend.ml.export_models` writes a JSON and an npz that reproduce the
    fitted estimators exactly. Reading those instead of the joblib files is
    what keeps scikit-learn, SciPy and their 180 MB out of the runtime.
    """
    chase_path = MODEL_DIR / f"{code}_chase.json"
    meta_path = MODEL_DIR / f"{code}_meta.json"
    if not chase_path.exists() or not meta_path.exists():
        raise LeagueNotAvailable(code)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chase_model = ChaseModel(json.loads(chase_path.read_text(encoding="utf-8")))

    setting_model = None
    setting_path = MODEL_DIR / f"{code}_setting.npz"
    if setting_path.exists():
        with np.load(setting_path, allow_pickle=False) as payload:
            setting_model = SettingModel({k: payload[k] for k in payload.files})

    return Bundle(
        code=code,
        meta=meta,
        chase_model=chase_model,
        setting_model=setting_model,
        priors=Priors.from_dict(meta.get("priors", {})),
    )


@lru_cache(maxsize=None)
def available_leagues() -> tuple:
    """League codes that have an exported model, in registry order."""
    codes = []
    for code in BY_CODE:
        if (MODEL_DIR / f"{code}_chase.json").exists():
            codes.append(code)
    return tuple(codes)


def resolve_league(code: str | None) -> str:
    available = available_leagues()
    if not available:
        raise LeagueNotAvailable("none")
    if code and code in available:
        return code
    if code:
        raise LeagueNotAvailable(code)
    return DEFAULT_LEAGUE if DEFAULT_LEAGUE in available else available[0]


# --------------------------------------------------------------------------
# core prediction
# --------------------------------------------------------------------------

def win_probability(bundle: Bundle, state: ChaseState) -> float:
    """Probability that the chasing side wins, with the settled cases short-circuited.

    A model is the wrong tool for an already-decided match: if the runs are
    scored the chase is won, and if the balls or the wickets are gone it is
    lost. Asking the classifier for those would produce a confident 0.97 where
    the truth is exactly 1.
    """
    total_balls = max(int(state.total_balls), 1)
    balls_bowled = min(max(int(state.balls_bowled), 0), total_balls)

    if state.score >= state.target:
        return 1.0
    if state.wickets_fallen >= 10 or balls_bowled >= total_balls:
        return 0.0

    state.venue_par = bundle.priors.par_for(state.venue)
    probability = float(bundle.chase_model.predict([chase_features(state)])[0])
    # Never show a certainty the model has not earned.
    return min(max(probability, 0.001), 0.999)


def project_score(bundle: Bundle, state: SettingState) -> dict | None:
    """Projected first-innings total, with a conformal range around it."""
    if bundle.setting_model is None:
        return None

    state.venue_par = bundle.priors.par_for(state.venue)
    midpoint = float(bundle.setting_model.predict([setting_features(state)])[0])
    band = bundle.setting_model.band(state.balls_bowled)

    # The projection can never fall below what is already on the board.
    floor = float(state.score)
    return {
        "projected": round(max(midpoint, floor)),
        "low": round(max(midpoint + band["low"], floor)),
        "high": round(max(midpoint + band["high"], floor + 1)),
        "confidence": bundle.meta.get("setting", {}).get("band_coverage"),
    }


# --------------------------------------------------------------------------
# derived reads on top of a probability
# --------------------------------------------------------------------------

def _overs_label(balls: int) -> str:
    return f"{balls // 6}.{balls % 6}"


def situation(state: ChaseState) -> dict:
    """The scoreboard numbers a viewer expects to see next to the percentage."""
    total_balls = max(int(state.total_balls), 1)
    balls_bowled = min(max(int(state.balls_bowled), 0), total_balls)
    balls_left = total_balls - balls_bowled
    runs_left = max(state.target - state.score, 0)
    wickets_left = max(10 - state.wickets_fallen, 0)

    crr = (state.score * 6 / balls_bowled) if balls_bowled else 0.0
    rrr = (runs_left * 6 / balls_left) if balls_left else 0.0

    return {
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "overs_label": _overs_label(balls_bowled),
        "overs_left_label": _overs_label(balls_left),
        "current_run_rate": round(crr, 2),
        "required_run_rate": round(rrr, 2) if balls_left else None,
        "phase": phase_of(balls_bowled, total_balls),
        "score_line": f"{state.score}/{state.wickets_fallen}",
    }


def confidence_of(probability: float) -> str:
    """How decisive the reading is, in words rather than a number."""
    distance = abs(probability - 0.5)
    if distance > 0.40:
        return "near certain"
    if distance > 0.28:
        return "strong"
    if distance > 0.15:
        return "clear"
    if distance > 0.06:
        return "slight"
    return "too close to call"


def narrate(state: ChaseState, probability: float, facts: dict) -> list:
    """Two or three sentences explaining what the number is reacting to."""
    lines = []
    batting, bowling = state.batting_team, state.bowling_team
    runs_left = facts["runs_left"]
    balls_left = facts["balls_left"]
    wickets_left = facts["wickets_left"]
    crr = facts["current_run_rate"]
    rrr = facts["required_run_rate"] or 0.0

    if probability >= 0.80:
        lines.append(f"{batting} are well on top of this chase.")
    elif probability >= 0.60:
        lines.append(f"{batting} hold the advantage, but it is not put away yet.")
    elif probability > 0.40:
        lines.append("This one is genuinely in the balance.")
    elif probability > 0.20:
        lines.append(f"{bowling} are in control, though the chase is not dead.")
    else:
        lines.append(f"{batting} need something extraordinary from here.")

    lines.append(
        f"{runs_left} needed from {balls_left} balls with {wickets_left} "
        f"{'wicket' if wickets_left == 1 else 'wickets'} in hand."
    )

    gap = rrr - crr
    if rrr and gap > 3:
        lines.append(
            f"The required rate has climbed to {rrr:.1f} against {crr:.1f} scored so far, "
            "so the asking is getting steeper every over."
        )
    elif rrr and gap < -1.5:
        lines.append(
            f"At {rrr:.1f} required against {crr:.1f} scored, there is room to take the "
            "singles and wait for the loose ball."
        )

    if wickets_left <= 2:
        lines.append("With the tail exposed, one more wicket could settle it.")
    elif wickets_left >= 8 and probability > 0.4:
        lines.append("Batting resources are barely touched, which is what keeps this alive.")

    if state.wickets_last_30 >= 3:
        lines.append(
            f"{int(state.wickets_last_30)} wickets have gone down in the last five overs."
        )
    elif state.runs_last_30 >= 55:
        lines.append(
            f"{int(state.runs_last_30)} runs have come in the last five overs, and that "
            "is where the momentum is."
        )

    return lines


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------

# The outcomes worth asking about for the next over, as (label, runs, wickets).
OVER_SCENARIOS = [
    ("Maiden, wicket", 0, 1),
    ("Quiet over", 3, 0),
    ("Par over", 7, 0),
    ("Par over, wicket", 6, 1),
    ("Good over", 12, 0),
    ("Big over", 18, 0),
    ("Big over, wicket", 16, 1),
]


def _advance(state: ChaseState, runs: int, wickets: int) -> ChaseState:
    """The same chase, one over later."""
    return ChaseState(
        batting_team=state.batting_team,
        bowling_team=state.bowling_team,
        target=state.target,
        score=state.score + runs,
        balls_bowled=min(state.balls_bowled + 6, state.total_balls),
        wickets_fallen=min(state.wickets_fallen + wickets, 10),
        total_balls=state.total_balls,
        venue=state.venue,
        runs_last_30=min(state.runs_last_30 + runs, 200) if state.balls_bowled >= 24
        else state.runs_last_30 + runs,
        wickets_last_30=state.wickets_last_30 + wickets,
    )


def next_over_scenarios(bundle: Bundle, state: ChaseState, current: float) -> list:
    """What each plausible next over would do to the number."""
    results = []
    for label, runs, wickets in OVER_SCENARIOS:
        if state.balls_bowled + 6 > state.total_balls and state.total_balls - state.balls_bowled > 0:
            # Fewer than six balls left; scale the scenario to what remains.
            remaining = state.total_balls - state.balls_bowled
            runs = round(runs * remaining / 6)
        after = _advance(state, runs, wickets)
        probability = win_probability(bundle, after)
        results.append({
            "label": label,
            "runs": runs,
            "wickets": wickets,
            "probability": round(probability, 4),
            "delta": round(probability - current, 4),
        })
    return results


def par_score_curve(bundle: Bundle, state: ChaseState, points: int = 42) -> list:
    """Win probability across the range of scores the side could be on right now.

    This is the chart that makes the model legible: it shows where the cliff
    is, which is far more informative than a single percentage.
    """
    ceiling = max(state.target, state.score + 1)
    step = max(1, ceiling // points)
    curve = []
    for score in range(0, ceiling + 1, step):
        probe = ChaseState(
            batting_team=state.batting_team,
            bowling_team=state.bowling_team,
            target=state.target,
            score=score,
            balls_bowled=state.balls_bowled,
            wickets_fallen=state.wickets_fallen,
            total_balls=state.total_balls,
            venue=state.venue,
            runs_last_30=state.runs_last_30,
            wickets_last_30=state.wickets_last_30,
        )
        curve.append({
            "score": score,
            "probability": round(win_probability(bundle, probe), 4),
        })
    return curve


def wickets_sensitivity(bundle: Bundle, state: ChaseState) -> list:
    """What the same score would be worth with more or fewer wickets standing."""
    out = []
    for fallen in range(0, 10):
        probe = ChaseState(
            batting_team=state.batting_team,
            bowling_team=state.bowling_team,
            target=state.target,
            score=state.score,
            balls_bowled=state.balls_bowled,
            wickets_fallen=fallen,
            total_balls=state.total_balls,
            venue=state.venue,
            runs_last_30=state.runs_last_30,
            wickets_last_30=state.wickets_last_30,
        )
        out.append({
            "wickets_left": 10 - fallen,
            "probability": round(win_probability(bundle, probe), 4),
        })
    return out


def required_matrix(state: ChaseState, overs: int = 6) -> list:
    """Runs needed per over if the chase is broken into the remaining overs."""
    balls_left = state.total_balls - state.balls_bowled
    runs_left = max(state.target - state.score, 0)
    if balls_left <= 0 or runs_left <= 0:
        return []

    rows = []
    remaining_overs = math.ceil(balls_left / 6)
    for index in range(min(overs, remaining_overs)):
        overs_after = remaining_overs - index - 1
        # Runs per over needed from here, if the earlier overs go exactly to plan.
        need_now = runs_left / (remaining_overs - index) if remaining_overs - index else runs_left
        rows.append({
            "over": state.balls_bowled // 6 + index + 1,
            "required_this_over": round(need_now, 1),
            "overs_after": overs_after,
        })
        runs_left = max(runs_left - round(need_now), 0)
    return rows
