"""
Real-match browsing and ball-by-ball replay.

Reads the processed Cricsheet tables and turns a stored match into the
win-probability graph you see on a broadcast: one reading per delivery, with
the moments that actually swung the game picked out.

The whole innings is scored in a single batched call rather than one per ball,
which is the difference between a replay that renders instantly and one that
takes several seconds.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.app.engine import FORM_WINDOW, Bundle, load_bundle
from backend.ml.features import CHASE_FEATURES, ChaseState, chase_features

DATA_DIR = Path("data/processed")

# A swing worth calling out, in probability points.
KEY_MOMENT_THRESHOLD = 0.06
MAX_KEY_MOMENTS = 8
# Even a procession has a passage where it was settled; always show a few.
MIN_KEY_MOMENTS = 4


class MatchNotFound(Exception):
    pass


@lru_cache(maxsize=None)
def _catalogue(code: str) -> pd.DataFrame:
    path = DATA_DIR / f"{code}_matches.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    return frame.sort_values("date", ascending=False).reset_index(drop=True)


@lru_cache(maxsize=None)
def _timeline(code: str) -> pd.DataFrame:
    path = DATA_DIR / f"{code}_timeline.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def warm(code: str) -> None:
    """Pull this league's tables into the cache ahead of the first request."""
    _catalogue(code)
    _timeline(code)


def available(code: str) -> bool:
    """Whether this league has the tables the browser and replay need."""
    return (DATA_DIR / f"{code}_timeline.parquet").exists()


def list_matches(
    code: str,
    search: str = "",
    season: str = "",
    limit: int = 40,
    offset: int = 0,
) -> dict:
    """Recent matches first, filtered by free text and season."""
    frame = _catalogue(code)
    if frame.empty:
        return {"total": 0, "matches": [], "seasons": []}

    seasons = sorted(frame["season"].dropna().astype(str).unique(), reverse=True)
    playable = frame[frame["replayable"]]

    if season:
        playable = playable[playable["season"].astype(str) == season]

    if search:
        needle = search.strip().lower()
        haystack = (
            playable["team_a"].fillna("") + " " + playable["team_b"].fillna("") + " "
            + playable["venue"].fillna("") + " " + playable["city"].fillna("") + " "
            + playable["date"].fillna("")
        ).str.lower()
        playable = playable[haystack.str.contains(needle, regex=False)]

    total = len(playable)
    window = playable.iloc[offset:offset + limit]

    return {
        "total": int(total),
        "seasons": seasons,
        "matches": [_summarise(row) for _, row in window.iterrows()],
    }


def _summarise(row) -> dict:
    margin = None
    if row["winner"] and pd.notna(row["second_score"]):
        if row["winner"] == row["second_team"]:
            margin = f"won by {10 - int(row['second_wickets'])} wickets"
        else:
            margin = f"won by {int(row['first_score']) - int(row['second_score'])} runs"

    return {
        "match_id": str(row["match_id"]),
        "date": str(row["date"]),
        "season": str(row["season"]),
        "venue": str(row["venue"]),
        "city": str(row["city"] or ""),
        "first_team": row["first_team"],
        "second_team": row["second_team"],
        "first_innings": (
            f"{int(row['first_score'])}/{int(row['first_wickets'])}"
            if pd.notna(row["first_score"]) else None
        ),
        "second_innings": (
            f"{int(row['second_score'])}/{int(row['second_wickets'])}"
            if pd.notna(row["second_score"]) else None
        ),
        "target": int(row["target"]) if pd.notna(row["target"]) else None,
        "winner": row["winner"],
        "margin": margin,
        "result": row["result"],
        "player_of_match": row["player_of_match"],
    }


def get_match(code: str, match_id: str) -> dict:
    frame = _catalogue(code)
    hit = frame[frame["match_id"].astype(str) == str(match_id)]
    if hit.empty:
        raise MatchNotFound(match_id)
    return _summarise(hit.iloc[0])


def replay(code: str, match_id: str) -> dict:
    """Ball-by-ball win probability for the chase, plus the moments that mattered."""
    bundle = load_bundle(code)
    frame = _catalogue(code)
    hit = frame[frame["match_id"].astype(str) == str(match_id)]
    if hit.empty:
        raise MatchNotFound(match_id)

    row = hit.iloc[0]
    if not row["replayable"]:
        raise MatchNotFound(f"{match_id} has no completed chase to replay")

    target = int(row["target"])
    total_balls = int(row["target_balls"]) if pd.notna(row["target_balls"]) else (
        int(row["scheduled_overs"]) * int(row["balls_per_over"])
    )
    batting = row["second_team"]
    bowling = row["first_team"]
    venue = row["venue"]

    deliveries = _timeline(code)
    chase = deliveries[
        (deliveries["match_id"].astype(str) == str(match_id))
        & (deliveries["innings"] == 2)
    ].sort_values(["over", "ball_in_over"])

    states, events = _walk_chase(chase, batting, bowling, target, total_balls, venue)
    probabilities = _score_all(bundle, states)

    balls = []
    previous = None
    for event, probability in zip(events, probabilities):
        event = dict(event)
        event["probability"] = round(probability, 4)
        event["delta"] = round(probability - previous, 4) if previous is not None else 0.0
        previous = probability
        balls.append(event)

    return {
        "match": _summarise(row),
        "batting_team": batting,
        "bowling_team": bowling,
        "target": target,
        "total_balls": total_balls,
        "balls": balls,
        "key_moments": _key_moments(balls),
    }


def _walk_chase(chase, batting, bowling, target, total_balls, venue) -> tuple:
    """Replay the innings, recording the state AFTER each delivery.

    A broadcast win-probability graph plots the state the match is in once a
    ball has been bowled, so the line and the scoreboard beside it always agree.
    The innings also gets a point at 0/0 before the first ball, which is what
    the opening move is measured against.
    """
    form_runs: deque = deque(maxlen=FORM_WINDOW)
    form_wickets: deque = deque(maxlen=FORM_WINDOW)

    start = ChaseState(
        batting_team=batting, bowling_team=bowling, target=target,
        score=0, balls_bowled=0, wickets_fallen=0,
        total_balls=total_balls, venue=venue,
    )
    states = [start]
    events = [{
        "over_label": "0.0",
        "ball_number": 0,
        "batter": None,
        "bowler": None,
        "runs": 0,
        "extra_kinds": "",
        "wicket": False,
        "wicket_kind": None,
        "player_out": None,
        "score": 0,
        "wickets": 0,
        "score_line": "0/0",
        "runs_left": target,
        "balls_left": total_balls,
        "start": True,
    }]

    legal = 0
    # itertuples is several times faster than iterrows, which matters because
    # this walks every delivery of the innings.
    for ball in chase.itertuples(index=False):
        if ball.legal:
            legal += 1
        form_runs.append(int(ball.runs_total))
        form_wickets.append(1 if ball.wicket else 0)

        score = int(ball.score)
        wickets = int(ball.wickets)

        states.append(ChaseState(
            batting_team=batting,
            bowling_team=bowling,
            target=target,
            score=score,
            balls_bowled=legal,
            wickets_fallen=wickets,
            total_balls=total_balls,
            venue=venue,
            runs_last_30=float(sum(form_runs)),
            wickets_last_30=float(sum(form_wickets)),
        ))
        events.append({
            "over_label": f"{int(ball.over)}.{int(ball.ball_in_over)}",
            "ball_number": legal,
            "batter": ball.batter,
            "bowler": ball.bowler,
            "runs": int(ball.runs_total),
            "extra_kinds": ball.extra_kinds,
            "wicket": bool(ball.wicket),
            "wicket_kind": ball.wicket_kind or None,
            "player_out": ball.player_out or None,
            "score": score,
            "wickets": wickets,
            "score_line": f"{score}/{wickets}",
            "runs_left": max(target - score, 0),
            "balls_left": max(total_balls - legal, 0),
            "start": False,
        })

    return states, events


def _score_all(bundle: Bundle, states: list) -> list:
    """One model call for the whole innings, with decided states filled in directly."""
    if not states:
        return []

    rows, indices = [], []
    probabilities = [0.0] * len(states)

    for index, state in enumerate(states):
        if state.score >= state.target:
            probabilities[index] = 1.0
        elif state.wickets_fallen >= 10 or state.balls_bowled >= state.total_balls:
            probabilities[index] = 0.0
        else:
            state.venue_par = bundle.priors.par_for(state.venue)
            rows.append(chase_features(state))
            indices.append(index)

    if rows:
        frame = pd.DataFrame(rows)[CHASE_FEATURES]
        predicted = bundle.chase_model.predict_proba(frame)[:, 1]
        for index, value in zip(indices, predicted):
            probabilities[index] = min(max(float(value), 0.001), 0.999)

    return probabilities


def _key_moments(balls: list) -> list:
    """The deliveries that moved the needle most, in chronological order.

    The cut-off adapts: a tight game produces plenty of six-point swings, while
    a one-sided one would return nothing at a fixed threshold even though it
    still had a passage where it was decided.
    """
    scored = [b for b in balls if not b.get("start")]
    if not scored:
        return []

    ranked = sorted(scored, key=lambda b: abs(b["delta"]), reverse=True)
    floor = max(KEY_MOMENT_THRESHOLD, 0.0)
    picked = [b for b in ranked if abs(b["delta"]) >= floor][:MAX_KEY_MOMENTS]
    if len(picked) < MIN_KEY_MOMENTS:
        picked = ranked[:MIN_KEY_MOMENTS]
    picked.sort(key=lambda b: b["ball_number"])

    moments = []
    for ball in picked:
        if ball["wicket"]:
            headline = f"{ball['player_out']} {ball['wicket_kind']}"
        elif ball["runs"] >= 6:
            headline = f"{ball['batter']} hits a six"
        elif ball["runs"] == 4:
            headline = f"{ball['batter']} finds the boundary"
        else:
            headline = f"{ball['runs']} off {ball['bowler']}"
        moments.append({**ball, "headline": headline})
    return moments


def team_list(code: str) -> list:
    """Teams, flagged by whether they still exist.

    Every league accumulates dead franchises - the IPL alone has five - and a
    picker that offers Deccan Chargers as readily as Chennai Super Kings is
    offering a team that folded in 2012. They stay available, because replaying
    an old match needs them, but they are grouped apart and never the default.
    """
    frame = _catalogue(code)
    if frame.empty:
        return []

    seasons = sorted(frame["season"].dropna().astype(str).unique())
    # "Recent" is the last four seasons, which is long enough to survive a
    # side missing a year and short enough to exclude the defunct.
    recent = set(seasons[-4:])
    live = frame[frame["season"].astype(str).isin(recent)]
    active = set(live["team_a"].dropna()) | set(live["team_b"].dropna())

    appearances = pd.concat([frame["team_a"], frame["team_b"]]).value_counts()
    last_seen = {}
    for column in ("team_a", "team_b"):
        for team, season in frame.groupby(column)["season"].max().items():
            last_seen[team] = max(str(season), last_seen.get(team, ""))

    teams = []
    for team, count in appearances.items():
        if not team:
            continue
        teams.append({
            "name": str(team),
            "active": str(team) in active,
            "matches": int(count),
            "last_season": last_seen.get(team, ""),
        })
    # Current sides first, then by how much history each one has.
    teams.sort(key=lambda t: (not t["active"], -t["matches"]))
    return teams


def venue_list(code: str) -> list:
    """Venues that have hosted matches, most used first."""
    frame = _catalogue(code)
    if frame.empty:
        return []
    seasons = sorted(frame["season"].dropna().astype(str).unique())
    recent = set(seasons[-6:])
    live = set(frame[frame["season"].astype(str).isin(recent)]["venue"].dropna())

    counts = frame["venue"].value_counts()
    venues = [
        {"venue": str(v), "matches": int(n), "active": str(v) in live}
        for v, n in counts.items()
    ]
    venues.sort(key=lambda v: (not v["active"], -v["matches"]))
    return venues
