"""
GameCastAI API.

A small JSON API in front of the trained models, plus the static frontend.
Everything the interface needs for one screen arrives in a single /predict
response, so the page never has to fan out into half a dozen requests while
someone is dragging a slider.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.app import matches as match_service
from backend.app.config import PROJECT_ROOT, public_config
from backend.app.engine import (
    LeagueNotAvailable,
    available_leagues,
    confidence_of,
    load_bundle,
    narrate,
    next_over_scenarios,
    par_score_curve,
    project_score,
    required_matrix,
    resolve_league,
    situation,
    win_probability,
    wickets_sensitivity,
)
from backend.app.schemas import ChaseRequest, SettingRequest
from backend.ml.features import ChaseState, SettingState
from backend.ml.leagues import BY_CODE

FRONTEND_DIR = PROJECT_ROOT / "frontend"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the default league before the first request rather than during it.

    A cold model load plus a cold parquet read costs about a second, and it
    would otherwise land on whoever happens to arrive first.
    """
    try:
        code = resolve_league(None)
        load_bundle(code)
        match_service.warm(code)
    except LeagueNotAvailable:
        # No models on disk yet. The API still starts and says so on /health,
        # which is friendlier than refusing to boot.
        pass
    yield


app = FastAPI(
    lifespan=lifespan,
    title="GameCastAI",
    description=(
        "Live T20 win probability and score projection, trained on real "
        "ball-by-ball data from Cricsheet."
    ),
    version="2.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# Replay payloads run to a few hundred kilobytes of JSON; compressing them
# makes the timeline usable on a phone connection.
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.exception_handler(LeagueNotAvailable)
async def _league_missing(_request, exc: LeagueNotAvailable):
    return JSONResponse(
        status_code=404,
        content={
            "detail": f"No trained model for league '{exc.args[0]}'.",
            "available": list(available_leagues()),
        },
    )


@app.exception_handler(match_service.MatchNotFound)
async def _match_missing(_request, exc: match_service.MatchNotFound):
    return JSONResponse(status_code=404, content={"detail": f"Match not found: {exc.args[0]}"})


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

@app.get("/api/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "leagues": list(available_leagues())}


@app.get("/api/leagues", tags=["meta"])
def leagues() -> dict:
    """Every league with a trained model, with enough detail to build the picker."""
    out = []
    for code in available_leagues():
        bundle = load_bundle(code)
        meta = bundle.meta
        league = BY_CODE[code]
        out.append({
            "code": code,
            "name": meta["league_name"],
            "short": meta["short"],
            "accent": league.accent,
            "blurb": league.blurb,
            "matches": meta["matches_total"],
            "date_from": meta["date_from"],
            "date_to": meta["date_to"],
            "seasons": meta["seasons"],
            "teams": meta["teams"],
            "accuracy": meta["chase"]["overall"]["accuracy"],
            "roc_auc": meta["chase"]["overall"]["roc_auc"],
        })
    return {"leagues": out, "default": resolve_league(None)}


@app.get("/api/leagues/{code}", tags=["meta"])
def league_detail(code: str) -> dict:
    """Teams, venues and the full model card for one league."""
    code = resolve_league(code)
    bundle = load_bundle(code)
    meta = bundle.meta
    venues = match_service.venue_list(code)
    teams = match_service.team_list(code)

    return {
        "code": code,
        "name": meta["league_name"],
        "short": meta["short"],
        "accent": BY_CODE[code].accent,
        "teams": [t["name"] for t in teams] or meta["teams"],
        "team_detail": teams,
        "venues": venues,
        "league_par": bundle.priors.league_par,
        "venue_par": bundle.priors.venue_par,
        "model_card": {
            "algorithm": meta["chase"]["algorithm"],
            "calibrated": meta["chase"]["calibrated"],
            "trained_at": meta["trained_at"],
            "data_source": meta["data_source"],
            "matches_total": meta["matches_total"],
            "date_from": meta["date_from"],
            "date_to": meta["date_to"],
            "seasons": meta["seasons"],
            "venues": meta["venues"],
            "train_matches": meta["chase"]["train_matches"],
            "test_matches": meta["chase"]["test_matches"],
            "shipped_matches": meta["chase"].get("shipped_matches"),
            "overall": meta["chase"]["overall"],
            "by_phase": meta["chase"]["by_phase"],
            "reliability": meta["chase"]["reliability"],
            "holdout_bias": meta["chase"].get("holdout_bias"),
            "max_calibration_gap": meta["chase"].get("max_calibration_gap"),
            "importances": meta["chase"]["importances"],
            "base_rate": meta["chase"]["base_rate"],
            "setting": meta.get("setting"),
        },
    }


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------

@app.post("/api/predict", tags=["predict"])
def predict(request: ChaseRequest) -> dict:
    """Everything the chase view needs, in one response."""
    code = resolve_league(request.league)
    bundle = load_bundle(code)

    state = ChaseState(
        batting_team=request.batting_team or "The chasing side",
        bowling_team=request.bowling_team or "The bowling side",
        target=request.target,
        score=request.score,
        balls_bowled=request.balls_bowled,
        wickets_fallen=request.wickets_fallen,
        total_balls=request.total_balls,
        venue=request.venue,
        runs_last_30=request.runs_last_30,
        wickets_last_30=request.wickets_last_30,
    )

    probability = win_probability(bundle, state)
    facts = situation(state)
    decided = (
        state.score >= state.target
        or state.wickets_fallen >= 10
        or state.balls_bowled >= state.total_balls
    )

    return {
        "league": code,
        "probability": round(probability, 4),
        "batting_probability": round(probability, 4),
        "bowling_probability": round(1 - probability, 4),
        "batting_team": state.batting_team,
        "bowling_team": state.bowling_team,
        "confidence": confidence_of(probability),
        "decided": decided,
        "situation": facts,
        "venue_par": bundle.priors.par_for(request.venue) if request.venue else None,
        "narrative": narrate(state, probability, facts),
        "scenarios": [] if decided else next_over_scenarios(bundle, state, probability),
        "score_curve": par_score_curve(bundle, state),
        "wickets_curve": wickets_sensitivity(bundle, state),
        "required_matrix": required_matrix(state),
    }


@app.post("/api/project", tags=["predict"])
def project(request: SettingRequest) -> dict:
    """First innings: what total is this side heading for?"""
    code = resolve_league(request.league)
    bundle = load_bundle(code)

    state = SettingState(
        batting_team=request.batting_team or "The batting side",
        bowling_team=request.bowling_team or "The bowling side",
        score=request.score,
        balls_bowled=request.balls_bowled,
        wickets_fallen=request.wickets_fallen,
        total_balls=request.total_balls,
        venue=request.venue,
        runs_last_30=request.runs_last_30,
        wickets_last_30=request.wickets_last_30,
    )

    projection = project_score(bundle, state)
    if projection is None:
        raise HTTPException(status_code=503, detail="No score model trained for this league.")

    balls_left = max(request.total_balls - request.balls_bowled, 0)
    par = bundle.priors.par_for(request.venue)
    return {
        "league": code,
        "projection": projection,
        "venue_par": par,
        "versus_par": round(projection["projected"] - par, 1),
        "situation": {
            "score_line": f"{request.score}/{request.wickets_fallen}",
            "balls_left": balls_left,
            "overs_label": f"{request.balls_bowled // 6}.{request.balls_bowled % 6}",
            "current_run_rate": round(
                request.score * 6 / request.balls_bowled, 2
            ) if request.balls_bowled else 0.0,
        },
    }


# --------------------------------------------------------------------------
# real matches
# --------------------------------------------------------------------------

@app.get("/api/matches", tags=["matches"])
def list_matches(
    league: str | None = None,
    search: str = Query(default="", max_length=80),
    season: str = Query(default="", max_length=16),
    limit: int = Query(default=40, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    code = resolve_league(league)
    payload = match_service.list_matches(code, search, season, limit, offset)
    return {"league": code, **payload}


@app.get("/api/matches/{match_id}", tags=["matches"])
def match_detail(match_id: str, league: str | None = None) -> dict:
    code = resolve_league(league)
    return {"league": code, "match": match_service.get_match(code, match_id)}


@app.get("/api/matches/{match_id}/replay", tags=["matches"])
def match_replay(match_id: str, league: str | None = None) -> dict:
    """Ball-by-ball win probability for a real chase."""
    code = resolve_league(league)
    return {"league": code, **match_service.replay(code, match_id)}


# --------------------------------------------------------------------------
# frontend
# --------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")
    app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon():
        return FileResponse(FRONTEND_DIR / "assets" / "favicon.svg")


@app.get("/config.js", include_in_schema=False)
def config_script() -> Response:
    """Hand the browser its settings as a script rather than a fetch.

    analytics.js has to know the measurement id before it decides whether to
    load the GA tag at all. A fetch would mean the tag arrives a round trip
    late and after the first paint, so instead this is a plain script tag that
    the browser executes in order, ahead of everything that reads it.

    Never cached: the whole point is that redeploying with a different
    environment variable takes effect without a rebuild.
    """
    payload = json.dumps(public_config(), separators=(",", ":"))
    body = f"window.GameCastConfig={payload};\n"
    return Response(
        content=body,
        media_type="application/javascript; charset=utf-8",
        headers={"Cache-Control": "no-store, max-age=0"},
    )
