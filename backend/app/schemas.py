"""Request and response shapes for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ChaseRequest(BaseModel):
    """A second-innings situation to score."""

    league: str | None = Field(default=None, description="League code, e.g. ipl")
    batting_team: str = Field(default="", max_length=80)
    bowling_team: str = Field(default="", max_length=80)
    venue: str = Field(default="", max_length=120)

    target: int = Field(ge=1, le=400, description="Runs needed to win")
    score: int = Field(ge=0, le=400)
    # Sent as balls rather than the 19.3 notation so there is one source of
    # truth; the interface does the conversion.
    balls_bowled: int = Field(ge=0, le=180)
    wickets_fallen: int = Field(ge=0, le=10)
    total_balls: int = Field(default=120, ge=6, le=180)

    runs_last_30: float = Field(default=0.0, ge=0, le=250)
    wickets_last_30: float = Field(default=0.0, ge=0, le=10)

    @model_validator(mode="after")
    def _coherent(self) -> "ChaseRequest":
        if self.balls_bowled > self.total_balls:
            raise ValueError("balls_bowled cannot exceed total_balls")
        if self.score >= self.target and self.score > 0:
            # Allowed, but it means the chase is already won; the engine
            # short-circuits rather than asking the model.
            pass
        return self


class SettingRequest(BaseModel):
    """A first-innings situation, where the question is the final total."""

    league: str | None = None
    batting_team: str = Field(default="", max_length=80)
    bowling_team: str = Field(default="", max_length=80)
    venue: str = Field(default="", max_length=120)

    score: int = Field(ge=0, le=400)
    balls_bowled: int = Field(ge=0, le=180)
    wickets_fallen: int = Field(ge=0, le=10)
    total_balls: int = Field(default=120, ge=6, le=180)

    runs_last_30: float = Field(default=0.0, ge=0, le=250)
    wickets_last_30: float = Field(default=0.0, ge=0, le=10)

    @model_validator(mode="after")
    def _coherent(self) -> "SettingRequest":
        if self.balls_bowled > self.total_balls:
            raise ValueError("balls_bowled cannot exceed total_balls")
        return self
