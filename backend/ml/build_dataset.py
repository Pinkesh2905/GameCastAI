"""
Build model-ready tables from the raw Cricsheet JSON.

Produces, per league, under data/processed/:

    {league}_chase.parquet     one row per delivery of a second innings,
                               labelled with what actually happened
    {league}_setting.parquet   first-innings rows labelled with the final total
    {league}_matches.parquet   a match catalogue for browsing and replay
    {league}_timeline.parquet  per-ball state, used to replay a real match

Two rules keep the labels honest:

  * every snapshot from a match carries the real result of that match, so the
    model learns cricket rather than the quirks of a generator, and
  * each row records the state BEFORE the delivery is bowled, so a row can
    never see the outcome of the ball it is being asked about.

    python -m backend.ml.build_dataset --league ipl
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import pandas as pd

from backend.ml.console import bullet, progress, say, step
from backend.ml.features import (
    ChaseState,
    SettingState,
    chase_features,
    setting_features,
)
from backend.ml.leagues import LEAGUES, get
from backend.ml.parse import Match, parse_match

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

# Rolling-form window: five overs is the span commentators actually talk about.
FORM_WINDOW = 30


class _Form:
    """Runs and wickets over the last `window` legal deliveries."""

    def __init__(self, window: int = FORM_WINDOW):
        self.runs = deque(maxlen=window)
        self.wickets = deque(maxlen=window)

    def add(self, runs: int, wicket: bool) -> None:
        self.runs.append(runs)
        self.wickets.append(1 if wicket else 0)

    @property
    def totals(self) -> tuple:
        return float(sum(self.runs)), float(sum(self.wickets))


def _chase_rows(match: Match) -> list:
    """Second-innings snapshots labelled with the real result."""
    if match.result != "win" or match.winner is None:
        return []          # ties and abandoned games have no binary label
    if match.method:
        # A rain-revised target changes mid-innings, but Cricsheet only stores
        # the final one, so earlier snapshots would carry a target that was not
        # yet in force. Cleaner to leave these out of training.
        return []
    if len(match.innings) < 2:
        return []

    innings = match.innings[1]
    if innings.target_runs is None:
        return []

    total_balls = innings.target_balls or (match.scheduled_overs * match.balls_per_over)
    if total_balls <= 0:
        return []

    label = int(match.winner == innings.team)
    form = _Form()
    rows = []

    score = wickets = balls = 0
    for ball in innings.balls:
        if ball.legal:
            runs_last_30, wickets_last_30 = form.totals
            state = ChaseState(
                batting_team=innings.team,
                bowling_team=innings.opponent,
                target=innings.target_runs,
                score=score,
                balls_bowled=balls,
                wickets_fallen=wickets,
                total_balls=total_balls,
                venue=match.venue,
                runs_last_30=runs_last_30,
                wickets_last_30=wickets_last_30,
            )
            row = chase_features(state)
            row["win"] = label
            # Context columns. These are not model inputs; training uses them
            # to attach venue priors, and the API uses them for display.
            row["match_id"] = match.match_id
            row["season"] = match.season
            row["venue"] = match.venue
            row["batting_team"] = innings.team
            row["bowling_team"] = innings.opponent
            row["target"] = innings.target_runs
            rows.append(row)
            balls += 1

        score = ball.score
        wickets = ball.wickets
        form.add(ball.runs_total, ball.wicket)

    return rows


def _setting_rows(match: Match) -> list:
    """First-innings snapshots labelled with the total the side went on to post."""
    if not match.innings:
        return []

    innings = match.innings[0]
    total_balls = match.scheduled_overs * match.balls_per_over
    # A side bowled out early still posted its total; a rain-curtailed innings
    # did not get its full allocation, so projecting it to 120 balls is wrong.
    if innings.legal_balls < total_balls and innings.final_wickets < 10:
        return []

    form = _Form()
    rows = []
    score = wickets = balls = 0

    for ball in innings.balls:
        if ball.legal:
            runs_last_30, wickets_last_30 = form.totals
            state = SettingState(
                batting_team=innings.team,
                bowling_team=innings.opponent,
                score=score,
                balls_bowled=balls,
                wickets_fallen=wickets,
                total_balls=total_balls,
                venue=match.venue,
                runs_last_30=runs_last_30,
                wickets_last_30=wickets_last_30,
            )
            row = setting_features(state)
            row["final_score"] = innings.final_score
            row["match_id"] = match.match_id
            row["season"] = match.season
            row["venue"] = match.venue
            row["batting_team"] = innings.team
            row["bowling_team"] = innings.opponent
            rows.append(row)
            balls += 1

        score = ball.score
        wickets = ball.wickets
        form.add(ball.runs_total, ball.wicket)

    return rows


def _match_row(match: Match) -> dict:
    first = match.innings[0] if match.innings else None
    second = match.innings[1] if len(match.innings) > 1 else None
    return {
        "match_id": match.match_id,
        "league": match.league,
        "date": match.date,
        "season": match.season,
        "venue": match.venue,
        "city": match.city,
        "team_a": match.teams[0],
        "team_b": match.teams[1],
        "toss_winner": match.toss_winner,
        "toss_decision": match.toss_decision,
        "winner": match.winner,
        "result": match.result,
        "method": match.method,
        "player_of_match": match.player_of_match,
        "scheduled_overs": match.scheduled_overs,
        "balls_per_over": match.balls_per_over,
        "first_team": first.team if first else None,
        "first_score": first.final_score if first else None,
        "first_wickets": first.final_wickets if first else None,
        "first_balls": first.legal_balls if first else None,
        "second_team": second.team if second else None,
        "second_score": second.final_score if second else None,
        "second_wickets": second.final_wickets if second else None,
        "second_balls": second.legal_balls if second else None,
        "target": second.target_runs if second else None,
        "target_balls": second.target_balls if second else None,
        "replayable": bool(second and second.target_runs),
    }


def _timeline_rows(match: Match) -> list:
    """Per-delivery record used to replay a match ball by ball."""
    rows = []
    for innings_no, innings in enumerate(match.innings[:2], start=1):
        for ball in innings.balls:
            rows.append({
                "match_id": match.match_id,
                "innings": innings_no,
                "over": ball.over,
                "ball_in_over": ball.ball_in_over,
                "legal": ball.legal,
                "batter": ball.batter,
                "bowler": ball.bowler,
                "runs_total": ball.runs_total,
                "runs_batter": ball.runs_batter,
                "extras": ball.extras,
                "extra_kinds": ",".join(ball.extra_kinds),
                "wicket": ball.wicket,
                "wicket_kind": ball.wicket_kind or "",
                "player_out": ball.player_out or "",
                "score": ball.score,
                "wickets": ball.wickets,
                "balls_bowled": ball.balls_bowled,
            })
    return rows


def build_league(code: str) -> dict:
    league = get(code)
    source = RAW_DIR / league.code
    files = sorted(source.glob("*.json"))
    if not files:
        bullet(f"{league.short:5} no raw data, run download_data first")
        return {}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chase, setting, matches, timeline = [], [], [], []
    skipped = 0

    for index, path in enumerate(files, start=1):
        match = parse_match(path, league.code)
        if match is None:
            skipped += 1
        else:
            matches.append(_match_row(match))
            timeline.extend(_timeline_rows(match))
            chase.extend(_chase_rows(match))
            setting.extend(_setting_rows(match))
        if index % 25 == 0 or index == len(files):
            progress(index, len(files), f"{league.short} {index:,}/{len(files):,} matches")

    written = {}
    for name, rows in (
        ("chase", chase),
        ("setting", setting),
        ("matches", matches),
        ("timeline", timeline),
    ):
        frame = pd.DataFrame(rows)
        destination = OUT_DIR / f"{league.code}_{name}.parquet"
        frame.to_parquet(destination, index=False)
        written[name] = len(frame)

    bullet(
        f"{league.short:5} {written['matches']:,} matches | "
        f"{written['chase']:,} chase rows | {written['setting']:,} first-innings rows | "
        f"{written['timeline']:,} deliveries"
        + (f" | {skipped} unreadable" if skipped else "")
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Build datasets from Cricsheet JSON")
    parser.add_argument("--league", action="append", help="league code (repeatable)")
    args = parser.parse_args()

    codes = args.league or [lg.code for lg in LEAGUES if (RAW_DIR / lg.code).exists()]

    step("Building datasets")
    for code in codes:
        build_league(code)

    say("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
