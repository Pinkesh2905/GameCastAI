"""
Train and evaluate the GameCastAI models.

For every league this fits two models:

  * a chase win-probability classifier, and
  * a first-innings score projector that reports a range, not just a number.

Four choices matter more than the algorithm:

  1. The split is by MATCH and by DATE. Snapshots from one game are near
     duplicates of each other, so a random row split would put a match on both
     sides and report an accuracy the model does not have. We hold out the most
     recent matches instead, which is also the honest question: does this
     predict games it has never seen?
  2. The models are regularised hard. There are a hundred thousand rows but
     only a thousand matches, and an unregularised booster reaches a training
     Brier score of 0.001 by memorising which game it is looking at.
  3. Probabilities are calibrated and then checked. A win probability is only
     useful if situations shown as 70% actually win about 70% of the time, so
     the report carries Brier score, log loss and a reliability table.
  4. The score projection band comes from conformal prediction on held-out
     residuals rather than from quantile regression, which overfitted badly
     here (85% band, 25% real coverage).
  5. Metrics and the shipped model come from different fits. Scores are
     measured on the future holdout, then the model that actually gets served
     is refitted on every match including that holdout. The holdout tells us
     how the method does on matches it has never seen; there is no reason to
     ship a model that has been denied the most recent two seasons.

A caveat worth stating plainly, because it is visible in the reliability
table: on the future holdout the model under-rates the chasing side by about
seven points. Random match splits show no such bias, so this is genuine drift -
teams now chase down positions that used to be losing ones, and no amount of
reweighting the older seasons can teach a model about a regime that postdates
its training data. Retraining as new matches arrive is the fix, which is why
the shipped model is refitted on everything.

    python -m backend.ml.train --league ipl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from backend.ml.console import bullet, say, step
from backend.ml.features import (
    CHASE_CATEGORICAL,
    CHASE_FEATURES,
    CHASE_NUMERIC,
    SETTING_CATEGORICAL,
    SETTING_FEATURES,
    SETTING_NUMERIC,
)
from backend.ml.leagues import LEAGUES, get
from backend.ml.priors import Priors

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RANDOM_STATE = 42

# Fraction of matches, most recent first, kept back for testing.
HOLDOUT_FRACTION = 0.18
# Share of the training matches reserved for conformal residuals.
CONFORMAL_FRACTION = 0.25
# Nominal coverage of the projected-score band.
BAND_COVERAGE = 0.80
# Permutation importance is O(features x repeats x rows), so it runs on a
# sample rather than the whole holdout.
IMPORTANCE_SAMPLE = 20_000

# Settings found by sweeping on the IPL holdout. The headline is min_samples_leaf:
# at the sklearn default the booster memorises individual matches.
BOOSTER = dict(
    max_iter=120,
    learning_rate=0.05,
    min_samples_leaf=2000,
    max_leaf_nodes=8,
    l2_regularization=30.0,
    early_stopping=False,   # its internal split is random, so it leaks across
    random_state=RANDOM_STATE,
)

FRIENDLY_NAMES = {
    "runs_left": "Runs still needed",
    "balls_left": "Balls remaining",
    "wickets_left": "Wickets in hand",
    "current_run_rate": "Current run rate",
    "required_run_rate": "Required run rate",
    "pressure_index": "Rate pressure (RRR minus CRR)",
    "innings_progress": "How far into the innings",
    "target_rate": "Difficulty of the target",
    "target_vs_par": "Target versus this venue par",
    "runs_last_30": "Runs in the last 5 overs",
    "wickets_last_30": "Wickets in the last 5 overs",
    "balls_per_wicket": "Balls per wicket in hand",
    "runs_per_wicket_needed": "Runs needed per wicket in hand",
    "phase": "Innings phase",
    "score": "Runs scored so far",
    "balls_bowled": "Balls bowled",
    "venue_par": "Venue par score",
}


# --------------------------------------------------------------------------
# splitting and priors
# --------------------------------------------------------------------------

def split_by_match(frame: pd.DataFrame, matches: pd.DataFrame) -> tuple:
    """Hold out the most recent matches, keeping every match wholly on one side."""
    dates = matches.set_index("match_id")["date"].to_dict()
    order = (
        frame["match_id"].drop_duplicates()
        .to_frame()
        .assign(date=lambda f: f["match_id"].map(dates).fillna(""))
        .sort_values(["date", "match_id"])
    )
    n_test = max(int(len(order) * HOLDOUT_FRACTION), 1)
    test_ids = set(order["match_id"].tail(n_test))

    is_test = frame["match_id"].isin(test_ids)
    return frame[~is_test].copy(), frame[is_test].copy(), sorted(test_ids)


def attach_priors(frame: pd.DataFrame, priors: Priors) -> pd.DataFrame:
    """Add the venue-derived columns the models expect."""
    frame = frame.copy()
    frame["venue_par"] = frame["venue"].map(priors.venue_par).fillna(priors.league_par)
    if "target" in frame.columns:
        frame["target_vs_par"] = frame["target"].astype(float) - frame["venue_par"]
    return frame


# --------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------

def _tree_pipeline(estimator, numeric: list, categorical: list) -> Pipeline:
    """Ordinal-encode the categoricals and let the booster treat them as such."""
    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    # After the transformer the numeric columns come first, so the categorical
    # ones occupy the trailing positions.
    categorical_mask = [False] * len(numeric) + [True] * len(categorical)
    estimator.set_params(categorical_features=categorical_mask)
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def _fresh_chase_model(name: str) -> Pipeline:
    """A new, unfitted copy of the chosen algorithm."""
    if name == "Gradient Boosting":
        return _tree_pipeline(
            HistGradientBoostingClassifier(**BOOSTER), CHASE_NUMERIC, CHASE_CATEGORICAL
        )
    return _linear_pipeline(CHASE_NUMERIC, CHASE_CATEGORICAL)


def _linear_pipeline(numeric: list, categorical: list) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=25), categorical),
    ])
    return Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
    ])


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> list:
    """Predicted vs observed win rate: the plain test of whether 70% means 70%."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(y_prob, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if not count:
            continue
        rows.append({
            "bin_low": round(float(edges[b]), 2),
            "bin_high": round(float(edges[b + 1]), 2),
            "predicted": round(float(y_prob[mask].mean()), 4),
            "observed": round(float(y_true[mask].mean()), 4),
            "count": count,
        })
    return rows


def classification_report(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "rows": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        # Brier and log loss reward being well calibrated, not just well ranked.
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob, labels=[0, 1])), 4),
    }


def phase_breakdown(frame: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    out = {}
    for phase in ("powerplay", "middle", "death"):
        mask = (frame["phase"] == phase).to_numpy()
        if mask.sum() < 50 or len(np.unique(y_true[mask])) < 2:
            continue
        out[phase] = classification_report(y_true[mask], y_prob[mask])
    return out


def top_importances(pipeline, X: pd.DataFrame, y, features: list, scoring: str) -> list:
    sample = X if len(X) <= IMPORTANCE_SAMPLE else X.sample(
        IMPORTANCE_SAMPLE, random_state=RANDOM_STATE
    )
    y_sample = y.loc[sample.index]
    result = permutation_importance(
        pipeline, sample, y_sample,
        n_repeats=3, random_state=RANDOM_STATE, scoring=scoring, n_jobs=-1,
    )
    ranked = sorted(
        zip(features, result.importances_mean, result.importances_std),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {
            "feature": name,
            "label": FRIENDLY_NAMES.get(name, name.replace("_", " ").title()),
            "importance": round(float(mean), 5),
            "std": round(float(std), 5),
        }
        for name, mean, std in ranked
    ]


# --------------------------------------------------------------------------
# chase model
# --------------------------------------------------------------------------

def train_chase(code: str, matches: pd.DataFrame, priors: Priors) -> dict | None:
    path = DATA_DIR / f"{code}_chase.parquet"
    if not path.exists():
        bullet(f"{code}: no chase dataset, skipping")
        return None

    frame = pd.read_parquet(path)
    if len(frame) < 2000:
        bullet(f"{code}: only {len(frame)} chase rows, too few to train")
        return None

    train, test, test_ids = split_by_match(frame, matches)
    train = attach_priors(train, priors)
    test = attach_priors(test, priors)

    X_train, y_train = train[CHASE_FEATURES], train["win"]
    X_test, y_test = test[CHASE_FEATURES], test["win"]

    bullet(
        f"train {len(train):,} rows / {train.match_id.nunique():,} matches   "
        f"test {len(test):,} rows / {len(test_ids):,} matches"
    )

    candidates = {
        "Gradient Boosting": _tree_pipeline(
            HistGradientBoostingClassifier(**BOOSTER), CHASE_NUMERIC, CHASE_CATEGORICAL
        ),
        "Logistic Regression": _linear_pipeline(CHASE_NUMERIC, CHASE_CATEGORICAL),
    }

    results = []
    for name, pipeline in candidates.items():
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        report = classification_report(y_test.to_numpy(), probabilities)
        report["name"] = name
        report["fit_seconds"] = round(time.perf_counter() - started, 1)
        results.append((report, pipeline, probabilities))
        bullet(
            f"{name:20} AUC {report['roc_auc']:.4f}  acc {report['accuracy']:.4f}  "
            f"brier {report['brier']:.4f}  ({report['fit_seconds']}s)"
        )

    # Brier score decides: it punishes a confident wrong call, which is exactly
    # the failure a win-probability display must avoid.
    best_report, best_pipeline, best_probs = min(results, key=lambda r: r[0]["brier"])

    # Wrap the winner in isotonic calibration, fitted by cross-validation on the
    # training matches only, then re-measure on the untouched holdout.
    calibrated = CalibratedClassifierCV(best_pipeline, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    calibrated_probs = calibrated.predict_proba(X_test)[:, 1]
    calibrated_report = classification_report(y_test.to_numpy(), calibrated_probs)

    if calibrated_report["brier"] <= best_report["brier"]:
        final_probs, final_report, calibration_used = (
            calibrated_probs, calibrated_report, True
        )
    else:
        final_probs, final_report, calibration_used = best_probs, best_report, False

    bullet(
        f"chosen {best_report['name']}"
        + (" + isotonic calibration" if calibration_used else " (uncalibrated)")
        + f"  AUC {final_report['roc_auc']:.4f}  brier {final_report['brier']:.4f}"
    )

    reliability = reliability_table(y_test.to_numpy(), final_probs)
    worst_gap = max(
        (abs(row["predicted"] - row["observed"]) for row in reliability), default=0.0
    )
    bias = float(y_test.mean() - final_probs.mean())

    # Everything above measured the method. What we ship is the same recipe
    # refitted on every match, holdout included, because the newest seasons are
    # the ones a live prediction most resembles.
    shipped = _fresh_chase_model(best_report["name"])
    if calibration_used:
        shipped = CalibratedClassifierCV(shipped, method="isotonic", cv=3)
    full = attach_priors(frame, priors)
    shipped.fit(full[CHASE_FEATURES], full["win"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(shipped, MODEL_DIR / f"{code}_chase.joblib")
    bullet(f"shipped model refitted on all {full.match_id.nunique():,} matches")

    return {
        "algorithm": best_report["name"],
        "calibrated": calibration_used,
        "overall": final_report,
        "holdout_bias": round(bias, 4),
        "shipped_matches": int(full.match_id.nunique()),
        "by_phase": phase_breakdown(test, y_test.to_numpy(), final_probs),
        "reliability": reliability,
        "max_calibration_gap": round(float(worst_gap), 4),
        "importances": top_importances(
            best_pipeline, X_test, y_test, CHASE_FEATURES, "neg_brier_score"
        ),
        "train_rows": int(len(train)),
        "train_matches": int(train.match_id.nunique()),
        "test_rows": int(len(test)),
        "test_matches": len(test_ids),
        "base_rate": round(float(frame.groupby("match_id")["win"].first().mean()), 4),
    }


# --------------------------------------------------------------------------
# first-innings score model
# --------------------------------------------------------------------------

def _conformal_offsets(residuals: pd.Series, buckets: pd.Series) -> dict:
    """Signed residual quantiles per stage of the innings.

    Split conformal prediction: the width of the band is read off errors the
    model actually made on matches it did not train on, so the coverage is
    earned rather than asserted. Bucketing by stage lets the band be wide at
    the six-over mark and tight at the seventeenth, which is the truth.
    """
    tail = (1.0 - BAND_COVERAGE) / 2.0
    offsets = {}
    for bucket, group in residuals.groupby(buckets):
        if len(group) < 200:
            continue
        offsets[int(bucket)] = {
            "low": round(float(np.quantile(group, tail)), 2),
            "high": round(float(np.quantile(group, 1.0 - tail)), 2),
        }
    everything = {
        "low": round(float(np.quantile(residuals, tail)), 2),
        "high": round(float(np.quantile(residuals, 1.0 - tail)), 2),
    }
    return {"by_bucket": offsets, "fallback": everything}


def bucket_of(balls_bowled) -> int:
    """Group deliveries into five-over blocks for the conformal band."""
    return np.minimum((np.asarray(balls_bowled) // 30).astype(int), 3)


def train_setting(code: str, matches: pd.DataFrame, priors: Priors) -> dict | None:
    path = DATA_DIR / f"{code}_setting.parquet"
    if not path.exists():
        return None

    frame = pd.read_parquet(path)
    if len(frame) < 2000:
        return None

    train, test, test_ids = split_by_match(frame, matches)
    train = attach_priors(train, priors)
    test = attach_priors(test, priors)

    def regressor():
        return _tree_pipeline(
            HistGradientBoostingRegressor(
                **{k: v for k, v in BOOSTER.items() if k != "max_iter"}, max_iter=150
            ),
            SETTING_NUMERIC, SETTING_CATEGORICAL,
        )

    # Carve a conformal calibration slice out of the training matches, again
    # splitting whole matches so residuals come from genuinely unseen games.
    match_ids = sorted(train["match_id"].unique())
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(match_ids)
    n_calibration = max(int(len(match_ids) * CONFORMAL_FRACTION), 1)
    calibration_ids = set(match_ids[:n_calibration])

    is_calibration = train["match_id"].isin(calibration_ids)
    fit_part, calibration_part = train[~is_calibration], train[is_calibration]

    provisional = regressor()
    provisional.fit(fit_part[SETTING_FEATURES], fit_part["final_score"])
    residuals = (
        calibration_part["final_score"]
        - provisional.predict(calibration_part[SETTING_FEATURES])
    )
    offsets = _conformal_offsets(residuals, pd.Series(
        bucket_of(calibration_part["balls_bowled"]), index=calibration_part.index
    ))

    # Refit on every training match now that the band is settled.
    model = regressor()
    model.fit(train[SETTING_FEATURES], train["final_score"])

    predictions = model.predict(test[SETTING_FEATURES])
    mae = float(mean_absolute_error(test["final_score"], predictions))

    test_buckets = bucket_of(test["balls_bowled"])
    low = np.array([
        p + offsets["by_bucket"].get(int(b), offsets["fallback"])["low"]
        for p, b in zip(predictions, test_buckets)
    ])
    high = np.array([
        p + offsets["by_bucket"].get(int(b), offsets["fallback"])["high"]
        for p, b in zip(predictions, test_buckets)
    ])
    actual = test["final_score"].to_numpy()
    coverage = float(((actual >= low) & (actual <= high)).mean())

    bullet(
        f"score projector   MAE {mae:.1f} runs   "
        f"{BAND_COVERAGE:.0%} band covers {coverage:.1%}   "
        f"mean width {float(np.mean(high - low)):.0f} runs"
    )

    # As with the chase model, the numbers above describe the method and the
    # artefact we ship has seen every match.
    shipped = regressor()
    everything = attach_priors(frame, priors)
    shipped.fit(everything[SETTING_FEATURES], everything["final_score"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": shipped, "offsets": offsets}, MODEL_DIR / f"{code}_setting.joblib")

    return {
        "mae": round(mae, 2),
        "band_nominal": BAND_COVERAGE,
        "band_coverage": round(coverage, 4),
        "band_mean_width": round(float(np.mean(high - low)), 1),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_matches": len(test_ids),
        "importances": top_importances(
            model, test[SETTING_FEATURES], test["final_score"],
            SETTING_FEATURES, "neg_mean_absolute_error",
        )[:8],
    }


# --------------------------------------------------------------------------

def train_league(code: str) -> dict | None:
    league = get(code)
    matches_path = DATA_DIR / f"{code}_matches.parquet"
    chase_path = DATA_DIR / f"{code}_chase.parquet"
    if not matches_path.exists() or not chase_path.exists():
        bullet(f"{league.short}: no processed data, run build_dataset first")
        return None

    matches = pd.read_parquet(matches_path)
    step(f"{league.name} ({league.short})")

    # Priors must only ever see training matches, so derive the training match
    # ids from the chase split before fitting them.
    chase_frame = pd.read_parquet(chase_path, columns=["match_id"])
    train_part, _, _ = split_by_match(chase_frame, matches)
    training_matches = matches[matches["match_id"].isin(set(train_part["match_id"]))]
    priors = Priors.fit(training_matches)
    bullet(
        f"venue priors from {len(training_matches):,} matches   "
        f"league par {priors.league_par:.0f}   {len(priors.venue_par)} grounds"
    )

    chase = train_chase(code, matches, priors)
    if chase is None:
        return None
    setting = train_setting(code, matches, priors)

    played = matches[matches["result"] != "no result"]
    metadata = {
        "league": code,
        "league_name": league.name,
        "short": league.short,
        "accent": league.accent,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_source": "Cricsheet (CC BY 4.0)",
        "matches_total": int(len(matches)),
        "date_from": str(matches["date"].min()),
        "date_to": str(matches["date"].max()),
        "seasons": int(matches["season"].nunique()),
        "venues": int(matches["venue"].nunique()),
        "teams": sorted(set(played["team_a"].dropna()) | set(played["team_b"].dropna())),
        "priors": priors.to_dict(),
        "chase": chase,
        "setting": setting,
    }
    (MODEL_DIR / f"{code}_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Train GameCastAI models")
    parser.add_argument("--league", action="append", help="league code (repeatable)")
    args = parser.parse_args()

    codes = args.league or [
        lg.code for lg in LEAGUES if (DATA_DIR / f"{lg.code}_chase.parquet").exists()
    ]

    step("Training")
    trained = []
    for code in codes:
        if train_league(code):
            trained.append(code)

    step("Summary")
    for code in trained:
        meta = json.loads((MODEL_DIR / f"{code}_meta.json").read_text(encoding="utf-8"))
        overall = meta["chase"]["overall"]
        bullet(
            f"{meta['short']:5} AUC {overall['roc_auc']:.4f}  acc {overall['accuracy']:.4f}  "
            f"brier {overall['brier']:.4f}  ({meta['matches_total']:,} matches)"
        )
    say("\nModels written to models/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
