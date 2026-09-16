"""
Historical priors learned from past matches.

The one-hot version of `venue` was tried first and made the model worse: with
sixty grounds and a thousand matches the booster memorised which game it was
looking at. A single smoothed number per ground carries the real signal - some
pitches are simply worth twenty more runs than others - without giving the
model a way to identify the match.

Every prior is shrunk toward the league average, so a ground with four matches
of history barely moves off the mean while one with two hundred is trusted.
The priors are always fitted on training matches only and then applied to both
splits, so the holdout never informs its own features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Weight of the league-average prior, expressed in matches. A venue needs
# roughly this many games before its own history outweighs the league mean.
SHRINKAGE = 30.0


def _shrunk(total: float, count: float, prior: float, weight: float = SHRINKAGE) -> float:
    """Empirical-Bayes blend of an observed mean with the league mean."""
    return (total + prior * weight) / (count + weight)


@dataclass
class Priors:
    """Per-venue scoring history, plus the fallbacks for anything unseen."""

    venue_par: dict = field(default_factory=dict)
    league_par: float = 160.0

    def par_for(self, venue: str | None) -> float:
        """Par first-innings total at this ground, or the league average."""
        if not venue:
            return self.league_par
        return self.venue_par.get(venue, self.league_par)

    def to_dict(self) -> dict:
        return {"venue_par": self.venue_par, "league_par": self.league_par}

    @classmethod
    def from_dict(cls, payload: dict) -> "Priors":
        return cls(
            venue_par=dict(payload.get("venue_par", {})),
            league_par=float(payload.get("league_par", 160.0)),
        )

    @classmethod
    def fit(cls, matches) -> "Priors":
        """Learn venue pars from a match catalogue (training rows only)."""
        scored = matches[matches["first_score"].notna()]
        if scored.empty:
            return cls()

        league_par = float(scored["first_score"].mean())
        grouped = scored.groupby("venue")["first_score"].agg(["sum", "count"])
        venue_par = {
            str(venue): round(_shrunk(row["sum"], row["count"], league_par), 2)
            for venue, row in grouped.iterrows()
        }
        return cls(venue_par=venue_par, league_par=round(league_par, 2))
