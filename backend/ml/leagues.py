"""
League registry for GameCastAI.

Each entry maps a GameCastAI league code to its Cricsheet download slug and
the display metadata the API and frontend need. Adding a new T20 competition
is a matter of adding a row here and re-running the pipeline.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class League:
    code: str            # internal id, used in the API and model files
    name: str            # full display name
    short: str           # compact label for chips / mobile
    cricsheet: str       # Cricsheet archive slug
    accent: str          # brand colour used by the frontend
    blurb: str
    aliases: dict = field(default_factory=dict)  # historical team name -> current


# Franchises get renamed and relocated; Cricsheet keeps the name that was used
# at the time, so we fold the history into whatever the team is called today.
IPL_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Pune Warriors": "Pune Warriors India",
}

BBL_ALIASES: dict = {}

LEAGUES = [
    League(
        code="ipl",
        name="Indian Premier League",
        short="IPL",
        cricsheet="ipl_male_json",
        accent="#f0c040",
        blurb="India's franchise T20 league.",
        aliases=IPL_ALIASES,
    ),
    League(
        code="t20i",
        name="T20 Internationals",
        short="T20I",
        cricsheet="t20s_male_json",
        accent="#38bdf8",
        blurb="Men's international T20 cricket.",
    ),
    League(
        code="bbl",
        name="Big Bash League",
        short="BBL",
        cricsheet="bbl_male_json",
        accent="#22c55e",
        blurb="Australia's franchise T20 league.",
        aliases=BBL_ALIASES,
    ),
    League(
        code="psl",
        name="Pakistan Super League",
        short="PSL",
        cricsheet="psl_male_json",
        accent="#a78bfa",
        blurb="Pakistan's franchise T20 league.",
    ),
    League(
        code="wpl",
        name="Women's Premier League",
        short="WPL",
        cricsheet="wpl_female_json",
        accent="#f472b6",
        blurb="India's women's franchise T20 league.",
    ),
]

BY_CODE = {lg.code: lg for lg in LEAGUES}
DEFAULT_LEAGUE = "ipl"


def get(code: str) -> League:
    try:
        return BY_CODE[code]
    except KeyError:
        raise KeyError(
            f"Unknown league {code!r}. Known: {', '.join(BY_CODE)}"
        ) from None


def canonical_team(league_code: str, team: str) -> str:
    """Map a historical team name onto the name the league uses today."""
    return get(league_code).aliases.get(team, team)
