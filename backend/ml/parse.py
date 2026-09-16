"""
Turn a Cricsheet match JSON into plain Python structures.

Cricsheet is faithful to what actually happened, which means the parser has to
cope with rain-revised targets, super overs, retired batters, abandoned games
and innings whose over count the source itself flags as miscounted. Everything
awkward is handled here so the dataset builder stays readable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from backend.ml.leagues import canonical_team

# A batter who retires hurt can come back, so it costs the side no resource.
# A batter who retires out cannot, so it does.
NON_DISMISSALS = {"retired hurt"}


@dataclass
class Ball:
    over: int              # zero-based over number
    ball_in_over: int      # 1-based, counting only legal deliveries
    legal: bool
    batter: str
    bowler: str
    runs_total: int
    runs_batter: int
    extras: int
    extra_kinds: tuple
    wicket: bool
    wicket_kind: str | None
    player_out: str | None
    # State AFTER this delivery.
    score: int = 0
    wickets: int = 0
    balls_bowled: int = 0


@dataclass
class Innings:
    team: str
    opponent: str
    balls: list = field(default_factory=list)
    target_runs: int | None = None     # runs needed to win (chase only)
    target_balls: int | None = None    # deliveries available (DLS aware)
    final_score: int = 0
    final_wickets: int = 0
    legal_balls: int = 0


@dataclass
class Match:
    match_id: str
    league: str
    date: str
    season: str
    venue: str
    city: str
    teams: tuple
    toss_winner: str | None
    toss_decision: str | None
    winner: str | None
    result: str            # "win" | "tie" | "no result"
    method: str | None     # "D/L" when rain revised the target
    player_of_match: str | None
    balls_per_over: int
    scheduled_overs: int
    innings: list = field(default_factory=list)


def _dismissals(delivery: dict) -> list:
    return [
        w for w in delivery.get("wickets", [])
        if w.get("kind") not in NON_DISMISSALS
    ]


def _parse_innings(raw: dict, league: str, opponent: str) -> Innings:
    innings = Innings(
        team=canonical_team(league, raw["team"]),
        opponent=opponent,
    )

    target = raw.get("target") or {}
    if target:
        innings.target_runs = int(target["runs"])
        # Cricsheet gives revised overs; convert to deliveries later using
        # balls_per_over, which the caller knows.
        innings.target_balls = target.get("overs")

    score = 0
    wickets = 0
    legal = 0

    for over in raw.get("overs", []):
        over_no = int(over.get("over", 0))
        ball_in_over = 0
        for delivery in over.get("deliveries", []):
            extras = delivery.get("extras", {}) or {}
            # A wide or a no-ball has to be re-bowled, so it does not advance
            # the over. Byes and leg-byes do.
            is_legal = not ("wides" in extras or "noballs" in extras)

            runs = delivery.get("runs", {}) or {}
            runs_total = int(runs.get("total", 0))
            score += runs_total

            outs = _dismissals(delivery)
            wickets += len(outs)

            if is_legal:
                legal += 1
                ball_in_over += 1

            innings.balls.append(Ball(
                over=over_no,
                ball_in_over=ball_in_over if is_legal else ball_in_over + 1,
                legal=is_legal,
                batter=delivery.get("batter", ""),
                bowler=delivery.get("bowler", ""),
                runs_total=runs_total,
                runs_batter=int(runs.get("batter", 0)),
                extras=int(runs.get("extras", 0)),
                extra_kinds=tuple(sorted(extras.keys())),
                wicket=bool(outs),
                wicket_kind=outs[0].get("kind") if outs else None,
                player_out=outs[0].get("player_out") if outs else None,
                score=score,
                wickets=wickets,
                balls_bowled=legal,
            ))

    innings.final_score = score
    innings.final_wickets = wickets
    innings.legal_balls = legal
    return innings


def parse_match(path: Path, league: str) -> Match | None:
    """Read one Cricsheet file. Returns None for anything unusable."""
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    info = raw.get("info", {})
    teams = [canonical_team(league, t) for t in info.get("teams", [])]
    if len(teams) != 2:
        return None

    outcome = info.get("outcome", {}) or {}
    winner = outcome.get("winner")
    if winner:
        winner = canonical_team(league, winner)
        result = "win"
    elif outcome.get("eliminator"):
        # Tied, then decided by a super over. The main match was a genuine tie.
        result = "tie"
    else:
        result = (outcome.get("result") or "no result").lower()

    balls_per_over = int(info.get("balls_per_over", 6) or 6)
    dates = info.get("dates") or [""]

    # Super-over innings are a separate three-ball contest and would pollute
    # a model of a 20-over chase, so they never enter the innings list.
    raw_innings = [i for i in raw.get("innings", []) if not i.get("super_over")]
    if not raw_innings:
        return None

    match = Match(
        match_id=path.stem,
        league=league,
        date=str(dates[0]),
        season=str(info.get("season", "")),
        venue=info.get("venue") or "Unknown Venue",
        city=info.get("city") or "",
        teams=tuple(teams),
        toss_winner=canonical_team(league, info["toss"]["winner"]) if info.get("toss") else None,
        toss_decision=(info.get("toss") or {}).get("decision"),
        winner=winner,
        result=result,
        method=outcome.get("method"),
        player_of_match=(info.get("player_of_match") or [None])[0],
        balls_per_over=balls_per_over,
        scheduled_overs=int(info.get("overs", 20) or 20),
        innings=[],
    )

    for raw_inn in raw_innings:
        team = canonical_team(league, raw_inn["team"])
        opponent = next((t for t in teams if t != team), "")
        innings = _parse_innings(raw_inn, league, opponent)
        if innings.target_balls is not None:
            innings.target_balls = int(innings.target_balls) * balls_per_over
        match.innings.append(innings)

    return match
