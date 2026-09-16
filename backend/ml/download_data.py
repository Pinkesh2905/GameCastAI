"""
Fetch Cricsheet ball-by-ball archives.

Cricsheet publishes every match as a JSON file under a permissive licence
(CC BY 4.0). We download one zip per league and keep it on disk so the rest of
the pipeline never needs the network.

    python -m backend.ml.download_data                 # every registered league
    python -m backend.ml.download_data --league ipl
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from backend.ml.console import bullet, progress, say, step
from backend.ml.leagues import LEAGUES, get

BASE_URL = "https://cricsheet.org/downloads"
RAW_DIR = Path("data/raw")
USER_AGENT = "GameCastAI/2.0 (+https://github.com/Pinkesh2905)"


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with dest.open("wb") as handle:
            while chunk := response.read(256 * 1024):
                handle.write(chunk)
                downloaded += len(chunk)
                progress(downloaded, total, f"{downloaded/1e6:5.1f} MB")


def fetch_league(code: str, force: bool = False) -> Path:
    league = get(code)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    archive = RAW_DIR / f"{league.cricsheet}.zip"
    target = RAW_DIR / league.code

    if target.exists() and any(target.glob("*.json")) and not force:
        count = len(list(target.glob("*.json")))
        bullet(f"{league.short:5} already present ({count:,} matches), skipping")
        return target

    bullet(f"{league.short:5} downloading {league.cricsheet}.zip")
    _download(f"{BASE_URL}/{league.cricsheet}.zip", archive)

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if m.endswith(".json")]
        for member in members:
            # Cricsheet zips are flat, but be defensive about nested paths.
            name = Path(member).name
            with zf.open(member) as src, (target / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)

    archive.unlink()
    bullet(f"{league.short:5} extracted {len(members):,} matches into {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Cricsheet data")
    parser.add_argument("--league", action="append", help="league code (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    codes = args.league or [lg.code for lg in LEAGUES]

    step("Cricsheet download")
    bullet("data (c) cricsheet.org, licensed CC BY 4.0")
    for code in codes:
        fetch_league(code, force=args.force)

    say("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
