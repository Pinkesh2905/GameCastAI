"""
Real-match browsing and ball-by-ball replay.

Reads the artefacts `backend.ml.build_serving_data` writes - a JSON catalogue
and a columnar npz of deliveries - and turns a stored match into the
win-probability graph you see on a broadcast: one reading per ball, with the
moments that actually swung the game picked out.

No pandas here. The catalogue is a list of dicts and the deliveries are NumPy
columns sliced by a precomputed row range, which is both faster than filtering
a DataFrame and 160 MB lighter in the deployed bundle.

The whole innings is scored in a single batched call rather than one per ball,
which is the difference between a replay that renders instantly and one that
takes several seconds.
"""

from __future__ import annotations

import json
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np

from backend.app.config import PROJECT_ROOT
from backend.app.engine import FORM_WINDOW, Bundle, load_bundle
from backend.ml.features import ChaseState, chase_features

DATA_DIR = PROJECT_ROOT / "data" / "serve"

# A swing worth calling out, in probability points.
KEY_MOMENT_THRESHOLD = 0.06
MAX_KEY_MOMENTS = 8
# Even a procession has a passage where it was settled; always show a few.
MIN_KEY_MOMENTS = 4


class MatchNotFound(Exception):
    pass


@lru_cache(maxsize=None)
def _catalogue(code: str) -> dict:
    """Matches for a league, newest first, with a lookup by id."""
    path = DATA_DIR / f"{code}_matches.json"
    if not path.exists():
        return {"matches": [], "seasons": [], "by_id": {}, "haystack": []}

    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = payload["matches"]

    # Build the search text once, rather than rebuilding it on every keystroke.
    haystack = [
        " ".join(
            str(match.get(field) or "")
            for field in ("team_a", "team_b", "venue", "city", "date", "season")
        ).lower()
        for match in matches
    ]
    return {
        "matches": matches,
        "seasons": payload.get("seasons", []),
        "by_id": {str(m["match_id"]): m for m in matches},
        "haystack": haystack,
    }


@lru_cache(maxsize=None)
def _timeline(code: str) -> dict | None:
    path = DATA_DIR / f"{code}_timeline.npz"
    if not path.exists():
        return None

    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}

    ids = [str(x) for x in arrays["match_ids"]]
    arrays["ranges"] = {
        match_id: (int(start), int(end))
        for match_id, start, end in zip(ids, arrays["row_start"], arrays["row_end"])
    }
    return arrays


def warm(code: str) -> None:
    """Pull this league's tables into the cache ahead of the first request."""
    _catalogue(code)
    _timeline(code)


def available(code: str) -> bool:
    """Whether this league has the tables the browser and replay need."""
    return (DATA_DIR / f"{code}_timeline.npz").exists()


# --------------------------------------------------------------------------
# browsing
# --------------------------------------------------------------------------

def list_matches(
    code: str,
    search: str = "",
    season: str = "",
    limit: int = 40,
    offset: int = 0,
) -> dict:
    """Recent matches first, filtered by free text and season."""
    catalogue = _catalogue(code)
    if not catalogue["matches"]:
        return {"total": 0, "matches": [], "seasons": []}

    needle = search.strip().lower()
    hits = []
    for index, match in enumerate(catalogue["matches"]):
        if not match.get("replayable"):
            continue
        if season and str(match.get("season")) != season:
            continue
        if needle and needle not in catalogue["haystack"][index]:
            continue
        hits.append(match)

    window = hits[offset:offset + limit]
    return {
        "total": len(hits),
        "seasons": catalogue["seasons"],
        "matches": [_summarise(match) for match in window],
    }


def _summarise(match: dict) -> dict:
    margin = None
    winner = match.get("winner")
    second_score = match.get("second_score")
    if winner and second_score is not None:
        if winner == match.get("second_team"):
            margin = f"won by {10 - int(match['second_wickets'])} wickets"
        else:
            margin = f"won by {int(match['first_score']) - int(second_score)} runs"

    def score_line(runs, wickets):
        if runs is None or wickets is None:
            return None
        return f"{int(runs)}/{int(wickets)}"

    return {
        "match_id": str(match["match_id"]),
        "date": str(match.get("date") or ""),
        "season": str(match.get("season") or ""),
        "venue": str(match.get("venue") or ""),
        "city": str(match.get("city") or ""),
        "first_team": match.get("first_team"),
        "second_team": match.get("second_team"),
        "first_innings": score_line(match.get("first_score"), match.get("first_wickets")),
        "second_innings": score_line(second_score, match.get("second_wickets")),
        "target": int(match["target"]) if match.get("target") is not None else None,
        "winner": winner,
        "margin": margin,
        "result": match.get("result"),
        "player_of_match": match.get("player_of_match"),
    }


def get_match(code: str, match_id: str) -> dict:
    match = _catalogue(code)["by_id"].get(str(match_id))
    if match is None:
        raise MatchNotFound(match_id)
    return _summarise(match)


def team_list(code: str) -> list:
    """Teams, flagged by whether they still exist.

    Every league accumulates dead franchises - the IPL alone has five - and a
    picker that offers Deccan Chargers as readily as Chennai Super Kings is
    offering a team that folded in 2012. They stay available, because replaying
    an old match needs them, but they are grouped apart and never the default.
    """
    matches = _catalogue(code)["matches"]
    if not matches:
        return []

    seasons = sorted({str(m.get("season") or "") for m in matches})
    # "Recent" is the last four seasons: long enough to survive a side missing
    # a year, short enough to exclude the defunct.
    recent = set(seasons[-4:])

    appearances: dict = {}
    last_seen: dict = {}
    active: set = set()
    for match in matches:
        season = str(match.get("season") or "")
        for field in ("team_a", "team_b"):
            team = match.get(field)
            if not team:
                continue
            appearances[team] = appearances.get(team, 0) + 1
            if season > last_seen.get(team, ""):
                last_seen[team] = season
            if season in recent:
                active.add(team)

    teams = [
        {
            "name": str(team),
            "active": team in active,
            "matches": count,
            "last_season": last_seen.get(team, ""),
        }
        for team, count in appearances.items()
    ]
    # Current sides first, then by how much history each one has.
    teams.sort(key=lambda t: (not t["active"], -t["matches"]))
    return teams


def venue_list(code: str) -> list:
    """Venues that have hosted matches, current grounds first."""
    matches = _catalogue(code)["matches"]
    if not matches:
        return []

    seasons = sorted({str(m.get("season") or "") for m in matches})
    recent = set(seasons[-6:])

    counts: dict = {}
    live: set = set()
    for match in matches:
        venue = match.get("venue")
        if not venue:
            continue
        counts[venue] = counts.get(venue, 0) + 1
        if str(match.get("season") or "") in recent:
            live.add(venue)

    venues = [
        {"venue": str(v), "matches": n, "active": v in live}
        for v, n in counts.items()
    ]
    venues.sort(key=lambda v: (not v["active"], -v["matches"]))
    return venues


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

def replay(code: str, match_id: str) -> dict:
    """Ball-by-ball win probability for the chase, plus the moments that mattered."""
    bundle = load_bundle(code)
    match = _catalogue(code)["by_id"].get(str(match_id))
    if match is None:
        raise MatchNotFound(match_id)
    if not match.get("replayable"):
        raise MatchNotFound(f"{match_id} has no completed chase to replay")

    timeline = _timeline(code)
    if timeline is None or str(match_id) not in timeline["ranges"]:
        raise MatchNotFound(f"{match_id} has no stored deliveries")

    target = int(match["target"])
    total_balls = int(match["target_balls"]) if match.get("target_balls") else (
        int(match["scheduled_overs"]) * int(match["balls_per_over"])
    )
    batting = match["second_team"]
    bowling = match["first_team"]
    venue = match["venue"]

    start, end = timeline["ranges"][str(match_id)]
    states, events = _walk_chase(
        timeline, start, end, batting, bowling, target, total_balls, venue
    )
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
        "match": _summarise(match),
        "batting_team": batting,
        "bowling_team": bowling,
        "target": target,
        "total_balls": total_balls,
        "balls": balls,
        "key_moments": _key_moments(balls),
    }


def _walk_chase(timeline, start, end, batting, bowling, target, total_balls, venue) -> tuple:
    """Replay the innings, recording the state AFTER each delivery.

    A broadcast win-probability graph plots the state the match is in once a
    ball has been bowled, so the line and the scoreboard beside it always agree.
    The innings also gets a point at 0/0 before the first ball, which is what
    the opening move is measured against.
    """
    form_runs: deque = deque(maxlen=FORM_WINDOW)
    form_wickets: deque = deque(maxlen=FORM_WINDOW)

    states = [ChaseState(
        batting_team=batting, bowling_team=bowling, target=target,
        score=0, balls_bowled=0, wickets_fallen=0,
        total_balls=total_balls, venue=venue,
    )]
    events = [{
        "over_label": "0.0", "ball_number": 0,
        "batter": None, "bowler": None, "runs": 0, "extra_kinds": "",
        "wicket": False, "wicket_kind": None, "player_out": None,
        "score": 0, "wickets": 0, "score_line": "0/0",
        "runs_left": target, "balls_left": total_balls, "start": True,
    }]

    over = timeline["over"][start:end]
    ball_in_over = timeline["ball_in_over"][start:end]
    legal = timeline["legal"][start:end]
    runs_total = timeline["runs_total"][start:end]
    score_col = timeline["score"][start:end]
    wickets_col = timeline["wickets"][start:end]
    wicket_col = timeline["wicket"][start:end]

    # The string columns are integer codes into a shared vocabulary, so this
    # resolves a whole innings of names in one fancy-index each.
    batters = timeline["batter_levels"][timeline["batter"][start:end]]
    bowlers = timeline["bowler_levels"][timeline["bowler"][start:end]]
    outs = timeline["player_out_levels"][timeline["player_out"][start:end]]
    kinds = timeline["wicket_kind_levels"][timeline["wicket_kind"][start:end]]
    extras = timeline["extra_kinds_levels"][timeline["extra_kinds"][start:end]]

    balls_bowled = 0
    for index in range(end - start):
        if legal[index]:
            balls_bowled += 1
        form_runs.append(int(runs_total[index]))
        form_wickets.append(1 if wicket_col[index] else 0)

        score = int(score_col[index])
        wickets = int(wickets_col[index])

        states.append(ChaseState(
            batting_team=batting, bowling_team=bowling, target=target,
            score=score, balls_bowled=balls_bowled, wickets_fallen=wickets,
            total_balls=total_balls, venue=venue,
            runs_last_30=float(sum(form_runs)),
            wickets_last_30=float(sum(form_wickets)),
        ))
        events.append({
            "over_label": f"{int(over[index])}.{int(ball_in_over[index])}",
            "ball_number": balls_bowled,
            "batter": str(batters[index]) or None,
            "bowler": str(bowlers[index]) or None,
            "runs": int(runs_total[index]),
            "extra_kinds": str(extras[index]),
            "wicket": bool(wicket_col[index]),
            "wicket_kind": str(kinds[index]) or None,
            "player_out": str(outs[index]) or None,
            "score": score,
            "wickets": wickets,
            "score_line": f"{score}/{wickets}",
            "runs_left": max(target - score, 0),
            "balls_left": max(total_balls - balls_bowled, 0),
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
        predicted = bundle.chase_model.predict(rows)
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
    picked = [b for b in ranked if abs(b["delta"]) >= KEY_MOMENT_THRESHOLD][:MAX_KEY_MOMENTS]
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
