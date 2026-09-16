"""
Repack the processed tables into a form the server can read without pandas.

The training pipeline is happy with parquet, but reading parquet at request
time means shipping pyarrow (87 MB) and pandas (74 MB) into the deployed
bundle for what amounts to two lookups: list the matches, and fetch the
deliveries of one chase.

So this writes, per league, under data/serve/:

    {league}_matches.json      the catalogue, small enough to be plain JSON
    {league}_timeline.npz      the deliveries, as columnar NumPy arrays

Only the second innings is kept. The replay view scores a chase, and the
first innings is already summarised in the catalogue, so carrying it would
double the payload for a screen that never asks for it.

Strings are stored once in a vocabulary with an integer column pointing into
it. Player names repeat hundreds of times across an innings, and this turns
what would be the largest part of the file into two bytes a row.

    python -m backend.ml.build_serving_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backend.ml.console import bullet, say, step
from backend.ml.leagues import LEAGUES

PROCESSED_DIR = Path("data/processed")
SERVE_DIR = Path("data/serve")


def _vocabulary(values) -> tuple:
    """Turn a string column into a vocabulary plus integer indices."""
    text = values.fillna("").astype(str)
    levels = sorted(set(text))
    lookup = {level: position for position, level in enumerate(levels)}
    codes = np.fromiter((lookup[v] for v in text), dtype=np.int32, count=len(text))
    # A league can exceed 65,535 distinct names once every T20 nation is in,
    # so widen only when it has to rather than always.
    width = np.uint16 if len(levels) <= np.iinfo(np.uint16).max else np.int32
    return np.asarray(levels, dtype=object).astype(str), codes.astype(width)


def _clean(value):
    """JSON cannot hold NaN, and the frontend should get null instead."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_matches(code: str) -> int:
    source = PROCESSED_DIR / f"{code}_matches.parquet"
    if not source.exists():
        return 0

    frame = pd.read_parquet(source).sort_values("date", ascending=False)

    records = []
    for row in frame.to_dict("records"):
        records.append({key: _clean(value) for key, value in row.items()})

    payload = {
        "league": code,
        "seasons": sorted(
            {str(r["season"]) for r in records if r["season"] is not None},
            reverse=True,
        ),
        "matches": records,
    }
    destination = SERVE_DIR / f"{code}_matches.json"
    destination.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return len(records)


def build_timeline(code: str) -> int:
    source = PROCESSED_DIR / f"{code}_timeline.parquet"
    if not source.exists():
        return 0

    frame = pd.read_parquet(source)
    frame = frame[frame["innings"] == 2].copy()
    # Group every match into one contiguous block so a replay is a slice
    # rather than a scan.
    frame = frame.sort_values(["match_id", "over", "ball_in_over"], kind="stable")
    frame = frame.reset_index(drop=True)

    match_ids = frame["match_id"].astype(str).to_numpy()
    boundaries = np.flatnonzero(np.r_[True, match_ids[1:] != match_ids[:-1]])
    starts = boundaries
    ends = np.r_[boundaries[1:], len(match_ids)] if len(boundaries) else np.zeros(0, int)

    batter_levels, batter_codes = _vocabulary(frame["batter"])
    bowler_levels, bowler_codes = _vocabulary(frame["bowler"])
    out_levels, out_codes = _vocabulary(frame["player_out"])
    kind_levels, kind_codes = _vocabulary(frame["wicket_kind"])
    extras_levels, extras_codes = _vocabulary(frame["extra_kinds"])

    arrays = {
        "match_ids": np.asarray(match_ids[starts], dtype=object).astype(str),
        "row_start": starts.astype(np.int64),
        "row_end": ends.astype(np.int64),

        "over": frame["over"].to_numpy(np.uint8),
        "ball_in_over": frame["ball_in_over"].to_numpy(np.uint8),
        "legal": frame["legal"].to_numpy(bool),
        "runs_total": frame["runs_total"].to_numpy(np.uint8),
        # A chase can pass 255, so this one needs the wider type.
        "score": frame["score"].to_numpy(np.uint16),
        "wickets": frame["wickets"].to_numpy(np.uint8),
        "balls_bowled": frame["balls_bowled"].to_numpy(np.uint8),
        "wicket": frame["wicket"].to_numpy(bool),

        "batter": batter_codes, "batter_levels": batter_levels,
        "bowler": bowler_codes, "bowler_levels": bowler_levels,
        "player_out": out_codes, "player_out_levels": out_levels,
        "wicket_kind": kind_codes, "wicket_kind_levels": kind_levels,
        "extra_kinds": extras_codes, "extra_kinds_levels": extras_levels,
    }

    np.savez_compressed(SERVE_DIR / f"{code}_timeline.npz", **arrays)
    return len(frame)


def build_league(code: str) -> None:
    SERVE_DIR.mkdir(parents=True, exist_ok=True)

    matches = build_matches(code)
    if not matches:
        bullet(f"{code:5} no processed data, run build_dataset first")
        return
    deliveries = build_timeline(code)

    matches_kb = (SERVE_DIR / f"{code}_matches.json").stat().st_size / 1024
    timeline_kb = (SERVE_DIR / f"{code}_timeline.npz").stat().st_size / 1024
    bullet(
        f"{code:5} {matches:5,} matches ({matches_kb:6.0f} KB)   "
        f"{deliveries:7,} deliveries ({timeline_kb:6.0f} KB)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Repack data for NumPy-only serving")
    parser.add_argument("--league", action="append", help="league code (repeatable)")
    args = parser.parse_args()

    codes = args.league or [
        lg.code for lg in LEAGUES
        if (PROCESSED_DIR / f"{lg.code}_matches.parquet").exists()
    ]

    step("Repacking data for NumPy-only serving")
    for code in codes:
        build_league(code)

    total = sum(p.stat().st_size for p in SERVE_DIR.glob("*")) / 1e6
    say(f"\nServing data: {total:.1f} MB total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
