"""
API behaviour, including the cricket rules the model must never be asked about.

These run against the real trained models, so they are also a smoke test that
what is on disk still loads and produces sane numbers.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def chase(**overrides):
    body = dict(
        league="ipl",
        batting_team="Chennai Super Kings",
        bowling_team="Mumbai Indians",
        venue="Wankhede Stadium, Mumbai",
        target=181, score=95, balls_bowled=66, wickets_fallen=3,
        total_balls=120, runs_last_30=44, wickets_last_30=1,
    )
    body.update(overrides)
    return client.post("/api/predict", json=body)


class TestMeta:
    def test_health(self):
        payload = client.get("/api/health").json()
        assert payload["status"] == "ok"
        assert "ipl" in payload["leagues"]

    def test_leagues_listing(self):
        payload = client.get("/api/leagues").json()
        assert payload["leagues"]
        for league in payload["leagues"]:
            assert league["matches"] > 0
            assert 0.5 < league["roc_auc"] <= 1.0

    def test_league_detail_has_current_teams_first(self):
        detail = client.get("/api/leagues/ipl").json()
        assert detail["team_detail"][0]["active"] is True
        # Every defunct side sorts after every current one.
        flags = [t["active"] for t in detail["team_detail"]]
        assert flags == sorted(flags, reverse=True)

    def test_unknown_league_is_a_clean_404(self):
        response = client.get("/api/leagues/kabaddi")
        assert response.status_code == 404
        assert "available" in response.json()


class TestPredict:
    def test_returns_a_usable_payload(self):
        data = chase().json()
        assert 0 < data["probability"] < 1
        assert data["situation"]["runs_left"] == 86
        assert data["situation"]["balls_left"] == 54
        assert data["narrative"]
        assert data["scenarios"]
        assert data["score_curve"]

    def test_probabilities_sum_to_one(self):
        data = chase().json()
        assert data["batting_probability"] + data["bowling_probability"] == pytest.approx(1.0)

    # -- the cases where a model must not be consulted at all ---------------

    def test_target_reached_is_certain(self):
        data = chase(score=181).json()
        assert data["probability"] == 1.0
        assert data["decided"] is True

    def test_all_out_is_lost(self):
        data = chase(wickets_fallen=10).json()
        assert data["probability"] == 0.0
        assert data["decided"] is True

    def test_overs_gone_is_lost(self):
        data = chase(balls_bowled=120, score=150).json()
        assert data["probability"] == 0.0

    def test_decided_matches_offer_no_scenarios(self):
        assert chase(score=181).json()["scenarios"] == []

    # -- cricket sense ------------------------------------------------------

    def test_a_wicket_never_helps_the_chasing_side(self):
        base = chase().json()["probability"]
        worse = chase(wickets_fallen=4).json()["probability"]
        assert worse < base

    def test_more_runs_never_hurt(self):
        low = chase(score=80).json()["probability"]
        high = chase(score=120).json()["probability"]
        assert high > low

    def test_probability_rises_monotonically_with_score(self):
        curve = chase().json()["score_curve"]
        values = [point["probability"] for point in curve]
        # Allow a hair of noise, but the shape has to be a climb.
        assert all(b >= a - 0.02 for a, b in zip(values, values[1:]))
        assert values[-1] > values[0]

    def test_a_stiffer_target_is_harder(self):
        easy = chase(target=140).json()["probability"]
        hard = chase(target=220).json()["probability"]
        assert hard < easy

    def test_wicket_scenarios_cost_more_than_the_runs_they_bring(self):
        data = chase().json()
        by_label = {s["label"]: s for s in data["scenarios"]}
        assert by_label["Par over, wicket"]["delta"] < by_label["Par over"]["delta"]

    def test_a_rain_shortened_chase_is_handled(self):
        data = chase(total_balls=60, balls_bowled=30, target=90, score=45).json()
        assert data["situation"]["balls_left"] == 30
        assert 0 < data["probability"] < 1

    # -- validation ---------------------------------------------------------

    @pytest.mark.parametrize("bad", [
        {"balls_bowled": 200},          # more balls than the innings holds
        {"wickets_fallen": 11},
        {"target": 0},
        {"score": -5},
        {"total_balls": 0},
    ])
    def test_incoherent_input_is_rejected(self, bad):
        assert chase(**bad).status_code == 422


class TestProject:
    def test_projection_has_an_ordered_band(self):
        data = client.post("/api/project", json={
            "league": "ipl", "venue": "Wankhede Stadium, Mumbai",
            "score": 62, "balls_bowled": 36, "wickets_fallen": 1,
            "total_balls": 120, "runs_last_30": 48, "wickets_last_30": 1,
        }).json()
        p = data["projection"]
        assert p["low"] <= p["projected"] <= p["high"]

    def test_projection_never_falls_below_the_score_already_made(self):
        data = client.post("/api/project", json={
            "league": "ipl", "score": 240, "balls_bowled": 114,
            "wickets_fallen": 2, "total_balls": 120,
        }).json()
        assert data["projection"]["low"] >= 240


class TestMatches:
    def test_listing(self):
        data = client.get("/api/matches", params={"league": "ipl", "limit": 5}).json()
        assert data["total"] > 500
        assert len(data["matches"]) == 5
        assert data["seasons"]

    def test_search_filters(self):
        data = client.get("/api/matches", params={
            "league": "ipl", "search": "Chennai", "limit": 10,
        }).json()
        for match in data["matches"]:
            joined = " ".join(filter(None, [
                match["first_team"], match["second_team"], match["venue"], match["city"],
            ])).lower()
            assert "chennai" in joined

    def test_missing_match_is_a_clean_404(self):
        assert client.get("/api/matches/does-not-exist/replay").status_code == 404


@pytest.fixture(scope="module")
def replay():
    """One real chase, scored end to end. Shared because it is the slow call."""
    listing = client.get("/api/matches", params={"league": "ipl", "limit": 1}).json()
    match_id = listing["matches"][0]["match_id"]
    return client.get(f"/api/matches/{match_id}/replay", params={"league": "ipl"}).json()


class TestReplay:
    def test_starts_before_the_first_ball(self, replay):
        first = replay["balls"][0]
        assert first["start"] is True
        assert first["score"] == 0
        assert first["ball_number"] == 0

    def test_probability_and_scoreboard_describe_the_same_moment(self, replay):
        # Each row is the state AFTER its delivery, so a completed chase ends
        # at certainty rather than at a confident guess.
        last = replay["balls"][-1]
        if last["score"] >= replay["target"]:
            assert last["probability"] == 1.0

    def test_ball_numbers_never_go_backwards(self, replay):
        numbers = [b["ball_number"] for b in replay["balls"]]
        assert numbers == sorted(numbers)

    def test_key_moments_are_chronological(self, replay):
        order = [m["ball_number"] for m in replay["key_moments"]]
        assert order == sorted(order)
        assert replay["key_moments"]
