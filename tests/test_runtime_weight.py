"""
The serving runtime must stay light enough to deploy.

Vercel caps a serverless function at 250 MB unzipped. scikit-learn, SciPy,
pandas and pyarrow come to roughly 415 MB between them and are not needed to
*use* a fitted model, so the exported artefacts are read with NumPy alone.

Nothing about that is self-enforcing. One convenient `import pandas` in a
request path would put the bundle back over the limit, and the failure would
show up as a deploy error long after the commit that caused it. This test
exercises every endpoint in a subprocess and then asks which modules got
loaded, so the regression is caught here instead.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Anything here is a training-time dependency. If one appears at request time,
# either it was imported by mistake or a deliberate decision was reversed
# without updating this list.
FORBIDDEN = ["pandas", "sklearn", "scipy", "pyarrow", "joblib", "matplotlib"]

PROBE = textwrap.dedent(
    """
    import json, sys
    import backend.app.main
    from fastapi.testclient import TestClient

    with TestClient(backend.app.main.app) as client:
        client.get("/api/health")
        client.get("/api/leagues")
        client.get("/api/leagues/ipl")
        client.get("/config.js")
        client.post("/api/predict", json={
            "league": "ipl", "target": 181, "score": 95, "balls_bowled": 66,
            "wickets_fallen": 3, "total_balls": 120,
            "venue": "Wankhede Stadium, Mumbai",
        })
        client.post("/api/project", json={
            "league": "ipl", "score": 62, "balls_bowled": 36,
            "wickets_fallen": 1, "total_balls": 120,
        })
        listing = client.get("/api/matches", params={"league": "ipl", "limit": 1})
        match_id = listing.json()["matches"][0]["match_id"]
        client.get(f"/api/matches/{match_id}/replay", params={"league": "ipl"})

    print(json.dumps(sorted(m for m in %(forbidden)r if m in sys.modules)))
    """
) % {"forbidden": FORBIDDEN}


def test_no_training_dependency_is_imported_at_request_time():
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    loaded = __import__("json").loads(result.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        "These training-only packages were imported while serving requests: "
        f"{loaded}. That puts the deployed bundle back over Vercel's 250 MB "
        "limit. Read the exported artefacts through backend.app.inference "
        "instead of loading the joblib pipelines."
    )
