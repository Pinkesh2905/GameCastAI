# GameCastAI

Live win probability and score projection for T20 cricket, trained on **5,889 real
matches** of ball-by-ball data.

Set up any chase and get a calibrated percentage for the side batting second, see
where the cliff is across every possible score, play out the next over, or replay a
real match delivery by delivery and watch the number move.

| League | Matches | Span | Accuracy | ROC-AUC | Brier |
|---|---:|---|---:|---:|---:|
| Indian Premier League | 1,243 | 2008–2026 | 81.3% | 0.912 | 0.128 |
| T20 Internationals | 3,539 | 2005–2026 | 87.1% | 0.948 | 0.092 |
| Big Bash League | 662 | 2011–2026 | 77.3% | 0.877 | 0.152 |
| Pakistan Super League | 357 | 2016–2026 | 81.6% | 0.922 | 0.123 |
| Women's Premier League | 88 | 2023–2026 | 86.4% | 0.949 | 0.097 |

Every figure is measured on matches held out **by date** — the most recent 18%, which
the model never saw during training. See [How it is measured](#how-it-is-measured) for
why that distinction does most of the work.

---

## Quick start

```bash
git clone https://github.com/Pinkesh2905/GameCastAI.git
cd GameCastAI
python -m venv .venv
```

Activate it — PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Then install the runtime dependencies — four packages, because the models are
exported to a form NumPy can evaluate on its own:

```bash
pip install -r requirements.txt
```

The exported models and the match archive are committed, so you can run it
immediately. (To retrain, you also need `requirements-dev.txt` — see
[Rebuilding from source](#rebuilding-from-source).)

```bash
uvicorn backend.app.main:app --reload
```

Open <http://localhost:8000>. Interactive API docs are at `/api/docs`.

---

## What it does

**Chase** — the main view. Pick the teams, the ground, the target and where the chase
stands, and get a win probability that updates as you drag. Alongside it:

- a curve of win probability across *every* score the side could be on right now, so
  you can see exactly where the chase stops being a coin flip
- what each plausible next over is worth, from a wicket-maiden to eighteen off it —
  tap one and it plays out for real
- what the same position is worth with more or fewer wickets standing
- the over-by-over breakdown of what has to come from here

**1st innings** — before there is a chase to model, the question is what total the
batting side will post. Returns a projection and an honest range around it.

**Replay** — every completed chase in the archive, scored ball by ball by the same
model. Drag the worm to scrub through the match, or hit play. The deliveries that
moved the number most are picked out and are clickable.

**Model** — the full model card: how it was tested, whether 70% really means 70%,
where it is confident and where it is not, and what it actually leans on.

Any chase you set up is in the URL, so "Copy link to this scenario" produces a link
that opens on exactly that situation. No account, no database.

---

## How the prediction works

Nine numbers describe a chase, and the model sees only these:

| Feature | Why |
|---|---|
| `runs_left`, `balls_left`, `wickets_left` | The three resources of a chase |
| `current_run_rate`, `required_run_rate` | What has been managed against what is needed |
| `pressure_index` | How much harder it has to get than it has been |
| `innings_progress` | Ten required at the halfway mark is not ten required at the death |
| `target_rate` | Lets one model serve a 120-ball chase and a rain-reduced 60-ball one |
| `target_vs_par` | 180 is stiff at a slow ground and ordinary at a fast one |
| `runs_last_30`, `wickets_last_30` | Momentum over the last five overs |
| `balls_per_wicket`, `runs_per_wicket_needed` | Batting resources per wicket in hand |
| `phase` | Powerplay, middle or death, scaled to a shortened innings |

### What is deliberately missing

**Team identity.** Batting and bowling team were tested as inputs and **removed**,
because they made out-of-sample accuracy worse — IPL holdout ROC-AUC fell from 0.903
to 0.895 with them in. A T20 squad is rebuilt at every auction, so "Mumbai Indians" is
not a stable thing to learn from a thousand matches, and the model was mostly using it
to memorise which game it was looking at. The teams are still shown everywhere in the
interface; they just do not move the number.

**The venue, as a category.** One-hot encoding sixty grounds had the same problem. The
ground genuinely matters, so it survives as a single smoothed number: `venue_par`, the
average first-innings total there, shrunk toward the league mean so a ground with four
matches of history barely moves off it. That one feature is the fourth most important
in the model.

### Settled positions never reach the model

If the runs are scored the chase is won; if the balls or the wickets are gone it is
lost. Asking a classifier would return a confident 0.97 where the truth is exactly 1,
so those cases short-circuit.

---

## How it is measured

Three decisions matter more than the choice of algorithm.

**1. The split is by match and by date.** Consecutive deliveries in one game are near
duplicates. A random row split puts the same match on both sides and reports an
accuracy the model does not have — the previous version of this project claimed
**0.9888 ROC-AUC** on synthetic data that way. The honest figure, on real matches held
out by date, is **0.912**.

**2. The models are regularised hard.** There are 133,000 IPL rows but only 1,195
matches. Left at scikit-learn's defaults the booster reaches a *training* Brier score
of 0.001 by memorising match trajectories, and falls apart on anything new.

**3. Calibration is checked, not assumed.** Ranking matches correctly is not enough:
if the page says 70%, those situations have to win about seven times in ten. The model
card plots predicted against observed and the training run reports Brier score and log
loss alongside accuracy.

### A caveat the model card states openly

On the most recent seasons the model **under-rates the chasing side by about seven
points**. Random match splits show no such bias, so this is genuine drift — teams now
chase down positions that used to be losing ones, and no amount of reweighting older
seasons teaches a model about a regime that postdates its training data.

The fix is retraining as matches arrive, which is why **the shipped model is refitted
on every match including the holdout**. The numbers in the table above describe the
method; the model actually answering requests has seen more than the model those
numbers were measured on.

### The score projection range

The first-innings band comes from **conformal prediction**: the width is read off
errors the model made on matches it had not trained on, bucketed by how far the innings
has gone. The coverage is therefore earned rather than asserted — an 80% band holds the
real total 73–87% of the time depending on league.

Quantile regression was tried first and claimed an 85% band that held the truth **25%**
of the time. Same overfitting, less visible.

---

## Setting up Google Analytics

You do not need to write any code — one line changes.

### 1. Create a property

Go to [analytics.google.com](https://analytics.google.com) and sign in. If this is your
first property, the setup wizard walks you through it; otherwise click the gear icon
(**Admin**, bottom left) → **Create** → **Property**.

Give it a name (`GameCastAI`), pick your time zone and currency, and continue through
the business questions. Choose **Web** as the platform when asked.

### 2. Create a data stream

**Admin** → **Data streams** → **Add stream** → **Web**.

- **Website URL** — where you will deploy. Use your real domain; `localhost` is not
  accepted and does not need to be, because the code already skips local traffic.
- **Stream name** — anything, e.g. `GameCastAI web`.

Click **Create stream**. The panel that opens shows a **MEASUREMENT ID** in the top
right, of the form `G-XXXXXXXXXX`. Copy it.

### 3. Set it as an environment variable

The id is never committed. It is read from `GA_MEASUREMENT_ID` at runtime and
handed to the browser by `/config.js`, so rotating it is an environment change
and a restart rather than a rebuild.

**Locally:**

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
GA_MEASUREMENT_ID=G-XXXXXXXXXX
```

`.env` is gitignored. A real environment variable always beats the file, so
`GA_MEASUREMENT_ID=G-OTHER python -m uvicorn backend.app.main:app` works for a
one-off.

**On a host,** set it in the platform's own panel rather than uploading a file:

| Platform | Where |
|---|---|
| Vercel | Project → Settings → Environment Variables |
| Render | Service → Environment → Environment Variables |
| Railway | Service → Variables |
| Fly.io | `fly secrets set GA_MEASUREMENT_ID=G-XXXXXXXXXX` |

> With no id set, nothing is loaded, no cookie is set, and no request leaves
> the browser. Localhost never reports even with an id in place, so your own
> testing will not pollute the numbers. A malformed id (a lowercase `g-`, or a
> legacy `UA-` id) logs a warning at startup instead of silently never
> reporting — which is the failure mode that costs you an afternoon.

### 4. What you will already be collecting

GA4 records page views automatically. On top of that, GameCastAI sends seven custom
events, each chosen because it answers a question worth asking:

| Event | Fires when | Tells you |
|---|---|---|
| `predict_run` | someone scores a chase | is the core tool actually used? |
| `scenario_click` | a what-if over is played out | do people explore, or just look? |
| `replay_open` | a real match is opened | is the archive worth its 7 MB? |
| `replay_scrub` | the worm is dragged | is the signature feature discovered? |
| `share_copy` | a scenario link is copied | does it spread? |
| `view_change` | a section is switched | what do people come for? |
| `league_change` | a competition is switched | is multi-league earning its keep? |

Each carries some of `league`, `phase`, `probability` (bucketed to the nearest 5) and
`section`.

### 5. Making the parameters filterable

Custom parameters show up in GA4's realtime debug view straight away, but to use them
as **dimensions** in reports you have to register them once:

**Admin** → **Custom definitions** → **Create custom dimension**

| Dimension name | Scope | Event parameter |
|---|---|---|
| League | Event | `league` |
| Innings phase | Event | `phase` |
| Win probability | Event | `probability` |
| Section | Event | `section` |

Registration is not retroactive — data only flows into a dimension from the moment you
create it, so do this before you launch rather than after.

### 6. Reading it

- **Reports → Realtime** — confirm it works at all. Should populate within seconds.
- **Reports → Engagement → Events** — counts per event. If `predict_run` is high but
  `scenario_click` is near zero, people are reading the number and leaving.
- **Explore → Free form** — drag `League` in as a dimension against `predict_run` to
  see whether anyone actually uses the leagues beyond the default.

Ordinary reports lag by up to 24–48 hours; Realtime does not. If Realtime works and
reports look empty the next morning, wait another day before assuming something broke.

### If nothing shows up

1. Open the deployed site, then DevTools → **Network**, and filter for `gtag` — you
   should see a request to `googletagmanager.com`. If not, the ID is wrong or you are
   on localhost.
2. Check that the ID starts with `G-`. A `UA-` id is Universal Analytics, retired in
   2023, and will not work.
3. An ad blocker or privacy extension will block it. Test in a clean profile.
4. The code honours Global Privacy Control and Do Not Track. If your browser sets
   either, it will deliberately not report.

### Privacy

No personal data is collected: no names, no identifiers, no free text. Team and venue
values are public sporting facts and the situation numbers are the user's own
hypothetical. If you deploy this where GDPR applies, you still need a cookie notice —
GA4 sets cookies regardless of what the payload contains.

---

## Deploying

One FastAPI process serves both the API and the frontend, and the runtime is
deliberately small enough to fit a serverless function.

### Vercel

`vercel.json` and `api/index.py` are already in the repo, so:

```bash
vercel
```

Set `GA_MEASUREMENT_ID` under Project → Settings → Environment Variables, and
redeploy. That is the whole deployment.

### Why the runtime is NumPy-only

The obvious build does not fit. Vercel caps a serverless function at **250 MB
unzipped**, and the training stack is nowhere near that:

| Package | Size |
|---|---:|
| SciPy | 138 MB |
| pyarrow | 87 MB |
| pandas | 74 MB |
| NumPy | 57 MB |
| scikit-learn | 45 MB |
| FastAPI + pydantic | 14 MB |
| **Total** | **~415 MB** |

None of it is needed to *use* a fitted model. The chase model is a logistic
regression — a standardisation, a one-hot and a dot product. The score model is
150 small regression trees. So `backend/ml/export_models.py` writes them out as
JSON and npz, and `backend/app/inference.py` evaluates them in NumPy alone.
The export refuses to write anything that does not match scikit-learn to 1e-9,
and in practice it matches to around 1e-16.

The data went the same way: `backend/ml/build_serving_data.py` repacks the
parquet tables into a JSON catalogue and columnar npz, which drops pandas and
pyarrow too.

What that buys:

| | Before | After |
|---|---:|---:|
| Deployed dependencies | ~415 MB | **~75 MB** |
| Model artefacts | 916 KB | 152 KB |
| Match archive | 6.9 MB | 5.1 MB |
| Cold start | 2,220 ms | **840 ms** |
| Replay request | 1,170 ms cold / 26 ms warm | **4.5 ms** |

`tests/test_runtime_weight.py` exercises every endpoint in a subprocess and
fails if pandas, scikit-learn, SciPy, pyarrow or joblib ends up imported. That
is the only thing standing between one convenient import and a bundle that no
longer deploys, so do not delete it.

### Anywhere else

Render, Railway and Fly.io run the same command with no special configuration:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

Build command `pip install -r requirements.txt`. Free tiers usually sleep after
inactivity, so the first visitor after a quiet spell pays the cold start.

## Project layout

```text
backend/
  ml/                    the offline pipeline
    leagues.py           competition registry - add a league here
    download_data.py     fetch Cricsheet archives
    parse.py             Cricsheet JSON to Python, incl. every awkward case
    features.py          THE feature definitions, shared by training and serving
    priors.py            venue par scores, shrunk toward the league mean
    build_dataset.py     ball-by-ball to labelled snapshots
    train.py             fit, calibrate, evaluate, write the model card
    export_models.py     scikit-learn -> JSON/npz, verified to 1e-9
    build_serving_data.py  parquet -> JSON/npz, so the server needs no pandas
  app/                   the serving layer (NumPy only, no scikit-learn)
    main.py              FastAPI routes
    config.py            environment settings, incl. the GA measurement id
    inference.py         model evaluation in NumPy
    engine.py            prediction, scenarios, narrative
    matches.py           match browsing and ball-by-ball replay
    schemas.py           request validation
frontend/
  index.html             four views, one page
  css/styles.css         the design system
  js/analytics.js        Google Analytics - put your ID here
  js/api.js              fetch wrapper with request cancellation
  js/charts.js           hand-built SVG, including the worm
  js/app.js              application logic and URL state
api/index.py             Vercel entry point
models/                  model cards, exported artefacts, and the joblib originals
data/serve/              what the server reads: catalogue JSON and delivery npz
data/processed/          parquet intermediates, used only to rebuild the above
tests/                   63 tests
```

`features.py` is the file to be careful with. Training and serving both import it, so a
change there moves every prediction — which is the point, because the alternative is
two implementations that quietly drift apart.

---

## Tests

```bash
pytest
```

63 tests. Beyond the usual shape checks they assert that the thing behaves like
cricket: a wicket never helps the chasing side, more runs never hurt, the probability
curve rises monotonically with score, a stiffer target is harder, a wicket costs more
than the runs that came with it, and a settled match returns exactly 0 or 1 rather than
a confident guess. One test guards the deployment: it fails if a training-only
package is imported while serving a request.

---

## Design notes

The palette comes off a manual tin scoreboard at a cricket ground: a green-biased
near-black board, chalk numerals, amber bulbs behind them, and the oxidised red of a
leather ball. Amber therefore means the batting side and red means the bowling side,
everywhere, without exception — including the shading above and below the even-money
line on the worm.

Charts are hand-written SVG rather than a charting library. The worm is cricket's own
graph and has conventions no general-purpose library knows about: territory shaded
toward whoever is ahead of 50%, wickets as ticks along the floor, overs rather than
time on the x-axis. Each chart declares a wide and a narrow geometry and picks between
them from the width it is given, so a phone gets a portrait-leaning worm rather than a
100-pixel smear.

Everything is keyboard reachable. The worm is a `role="slider"` — arrow keys step a
ball, shift-arrow an over, Home and End jump to either end.

---

## Data and licence

Ball-by-ball data from [Cricsheet](https://cricsheet.org), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Rain-revised (D/L) matches
and super overs are excluded from training — Cricsheet stores only the final revised
target, so earlier deliveries in a D/L match would carry a target that was not yet in
force.

Not affiliated with any league, board or broadcaster. This is a modelling exercise, not
betting advice.
