"""
Vercel entry point.

Vercel looks for a module under `api/` and, for Python, serves whatever ASGI
application it finds as `app`. Everything real lives in `backend/app/main.py`;
this file exists only so the platform has something to point at.
"""

import sys
from pathlib import Path

# The function is unpacked into a directory that is not necessarily on the
# import path, so make the repository root importable before reaching for it.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import app  # noqa: E402

__all__ = ["app"]
