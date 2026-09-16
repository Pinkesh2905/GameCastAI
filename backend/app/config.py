"""
Runtime configuration, read from the environment.

Anything that differs between a laptop and a deployment lives here, and
nothing here is ever committed with a real value in it.

Locally, a `.env` file at the project root is read on import. Hosting
platforms - Vercel, Render, Railway, Fly - inject their own variables into the
process directly, so the file is simply absent there and `os.environ` already
holds the answer. That is why this does not use python-dotenv: the twenty
lines below are the whole of what is needed, and a dependency that only
matters on a developer machine is not worth shipping to a serverless bundle.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ENV_FILE = Path(".env")

# GA4 stream ids are always G- followed by an alphanumeric block. Validating
# the shape means a typo fails loudly here rather than silently never
# reporting, which is the failure mode that wastes an afternoon.
MEASUREMENT_ID_PATTERN = re.compile(r"^G-[A-Z0-9]{6,}$")


def _load_env_file(path: Path = ENV_FILE) -> None:
    """Read KEY=value lines into os.environ, without overriding what is set.

    A real environment variable always wins over the file, so exporting one
    for a single run does what you would expect.
    """
    if not path.exists():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


def ga_measurement_id() -> str:
    """The GA4 stream id, or an empty string when analytics is switched off.

    An empty value is a legitimate configuration, not an error: the frontend
    loads no tag at all and sets no cookie. A malformed value is different -
    somebody meant to enable it and mistyped - so that warns.
    """
    value = (os.environ.get("GA_MEASUREMENT_ID") or "").strip()
    if not value:
        return ""
    if not MEASUREMENT_ID_PATTERN.match(value):
        import warnings

        warnings.warn(
            f"GA_MEASUREMENT_ID={value!r} does not look like a GA4 stream id "
            "(expected G-XXXXXXXXXX). Analytics will stay off.",
            stacklevel=2,
        )
        return ""
    return value


def site_url() -> str:
    """Canonical public URL, used for link previews. Optional."""
    return (os.environ.get("SITE_URL") or "").strip().rstrip("/")


def public_config() -> dict:
    """Exactly the settings the browser is allowed to see.

    This is an allowlist on purpose. Everything returned here is served to
    every visitor, so it must never grow to include a secret by accident.
    """
    return {
        "gaMeasurementId": ga_measurement_id(),
        "siteUrl": site_url(),
    }
