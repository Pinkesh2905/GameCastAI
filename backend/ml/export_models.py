"""
Export the fitted scikit-learn models into a form pure NumPy can evaluate.

Why this exists: serving the pickled pipelines drags scikit-learn, SciPy,
pandas and pyarrow into the runtime, which is about 415 MB unzipped and does
not fit in a serverless function. None of it is needed to *use* these models.
The chase model is a logistic regression - a standardisation, a one-hot and a
dot product. The score model is 150 small regression trees. Both are a few
hundred lines of arithmetic, and NumPy alone does that in 57 MB.

Training still uses scikit-learn, and always will. This runs after it and
writes a second artefact beside each pickle:

    models/{league}_chase.json      scaler, coefficients, isotonic calibration
    models/{league}_setting.npz     tree arrays, thresholds, category bitsets

Both are verified against the original estimator before they are written, to
a tolerance tighter than anything the interface could display. If the export
ever stops matching, this fails rather than quietly serving a different model.

    python -m backend.ml.export_models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from backend.ml.console import bullet, say, step
from backend.ml.features import (
    CHASE_CATEGORICAL,
    CHASE_FEATURES,
    CHASE_NUMERIC,
    SETTING_CATEGORICAL,
    SETTING_FEATURES,
    SETTING_NUMERIC,
)
from backend.ml.leagues import LEAGUES

MODEL_DIR = Path("models")

# The exported model must agree with scikit-learn to far better than the
# interface can show. A displayed probability has one decimal place.
TOLERANCE = 1e-9
VERIFY_ROWS = 400
RANDOM_STATE = 7


# --------------------------------------------------------------------------
# pulling values out of a fitted pipeline
# --------------------------------------------------------------------------

def _pipeline_parts(pipeline) -> dict:
    """Scaler statistics and one-hot categories from a fitted chase pipeline."""
    prep = pipeline.named_steps["prep"]
    scaler = prep.named_transformers_["num"]
    encoder = prep.named_transformers_["cat"]
    model = pipeline.named_steps["model"]

    return {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        # OneHotEncoder(min_frequency=...) can fold rare levels into an
        # "infrequent" column, so read the categories it actually emitted
        # rather than assuming they match the input.
        "categories": [list(map(str, cats)) for cats in encoder.categories_],
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }


def _isotonic_parts(calibrator) -> dict:
    return {
        "x": calibrator.X_thresholds_.tolist(),
        "y": calibrator.y_thresholds_.tolist(),
    }


def export_chase(code: str) -> dict | None:
    source = MODEL_DIR / f"{code}_chase.joblib"
    if not source.exists():
        return None

    estimator = joblib.load(source)

    if isinstance(estimator, CalibratedClassifierCV):
        # cv=3 refits the pipeline on each fold and pairs it with its own
        # isotonic map. The prediction is the mean of the three.
        members = []
        for member in estimator.calibrated_classifiers_:
            part = _pipeline_parts(member.estimator)
            part["isotonic"] = _isotonic_parts(member.calibrators[0])
            members.append(part)
        payload = {"kind": "calibrated_logistic", "members": members}
    else:
        payload = {"kind": "logistic", "members": [_pipeline_parts(estimator)]}

    payload["numeric"] = CHASE_NUMERIC
    payload["categorical"] = CHASE_CATEGORICAL
    payload["feature_order"] = CHASE_FEATURES
    return payload


# --------------------------------------------------------------------------
# trees
# --------------------------------------------------------------------------

def export_setting(code: str) -> dict | None:
    """Flatten every tree into parallel arrays with an offset per tree."""
    source = MODEL_DIR / f"{code}_setting.joblib"
    if not source.exists():
        return None

    bundle = joblib.load(source)
    pipeline = bundle["model"]
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    encoder = prep.named_transformers_["cat"]

    # HistGradientBoosting does NOT see the columns in the order our own
    # ColumnTransformer emits them. When categorical_features is set it builds
    # a second, internal ColumnTransformer that hoists the categorical columns
    # to the front and re-encodes them. Read the real order off the fitted
    # estimator rather than assuming it, because assuming it produces a model
    # that is silently wrong by tens of runs.
    outer_columns = SETTING_NUMERIC + SETTING_CATEGORICAL
    mask = np.asarray(model.is_categorical_, dtype=bool)
    tree_order = (
        [outer_columns[i] for i in np.flatnonzero(mask)]
        + [outer_columns[i] for i in np.flatnonzero(~mask)]
    )

    # Two encoders run in series on the categorical column: ours, then the one
    # inside the estimator. Compose them into a single label -> value map, so
    # serving never has to replicate the sandwich. A label neither has seen
    # becomes NaN and takes the missing-value branch, as it does in sklearn.
    inner_encoder = model._preprocessor.named_transformers_["encoder"]
    phase_values = {}
    for position, label in enumerate(encoder.categories_[0]):
        inner = list(inner_encoder.categories_[0])
        phase_values[str(label)] = (
            float(inner.index(float(position))) if float(position) in inner else float("nan")
        )

    fields = (
        "value", "feature_idx", "num_threshold", "missing_go_to_left",
        "left", "right", "is_leaf", "is_categorical", "bitset_idx",
    )
    columns: dict = {name: [] for name in fields}
    offsets = [0]
    bitsets = []

    for stage in model._predictors:
        for predictor in stage:
            nodes = predictor.nodes
            for name in fields:
                columns[name].append(np.asarray(nodes[name]))
            # Each tree owns its bitsets, so shift the indices as trees are
            # concatenated into one flat table.
            raw = predictor.raw_left_cat_bitsets
            shift = len(bitsets)
            if shift:
                idx = columns["bitset_idx"][-1].copy()
                cat = columns["is_categorical"][-1].astype(bool)
                idx[cat] = idx[cat] + shift
                columns["bitset_idx"][-1] = idx
            bitsets.extend(np.asarray(raw).reshape(-1, 8))
            offsets.append(offsets[-1] + len(nodes))

    arrays = {
        "value": np.concatenate(columns["value"]).astype(np.float64),
        "feature_idx": np.concatenate(columns["feature_idx"]).astype(np.int32),
        "num_threshold": np.concatenate(columns["num_threshold"]).astype(np.float64),
        "missing_go_to_left": np.concatenate(columns["missing_go_to_left"]).astype(np.uint8),
        "left": np.concatenate(columns["left"]).astype(np.int32),
        "right": np.concatenate(columns["right"]).astype(np.int32),
        "is_leaf": np.concatenate(columns["is_leaf"]).astype(np.uint8),
        "is_categorical": np.concatenate(columns["is_categorical"]).astype(np.uint8),
        "bitset_idx": np.concatenate(columns["bitset_idx"]).astype(np.int32),
        "tree_offset": np.asarray(offsets, dtype=np.int64),
        "bitsets": (
            np.asarray(bitsets, dtype=np.uint32) if bitsets
            else np.zeros((0, 8), dtype=np.uint32)
        ),
        "baseline": np.asarray([float(np.ravel(model._baseline_prediction)[0])]),
        # The order the trees index into, categorical columns first.
        "tree_order": np.asarray(tree_order),
        "phase_labels": np.asarray(list(phase_values)),
        "phase_values": np.asarray(list(phase_values.values()), dtype=np.float64),
    }

    # The conformal band travels with the model; it is part of the answer.
    offsets_payload = bundle["offsets"]
    arrays["offsets_json"] = np.asarray([json.dumps(offsets_payload)])
    return arrays


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def _random_chase_frame(rng) -> pd.DataFrame:
    """Plausible but deliberately wide-ranging chase states."""
    n = VERIFY_ROWS
    rows = {
        "runs_left": rng.integers(0, 250, n).astype(float),
        "balls_left": rng.integers(0, 121, n).astype(float),
        "wickets_left": rng.integers(0, 11, n).astype(float),
        "current_run_rate": rng.uniform(0, 20, n),
        "required_run_rate": rng.uniform(0, 36, n),
        "pressure_index": rng.uniform(-10, 20, n),
        "innings_progress": rng.uniform(0, 1, n),
        "target_rate": rng.uniform(4, 14, n),
        "target_vs_par": rng.uniform(-60, 90, n),
        "runs_last_30": rng.uniform(0, 110, n),
        "wickets_last_30": rng.integers(0, 8, n).astype(float),
        "balls_per_wicket": rng.uniform(0, 60, n),
        "runs_per_wicket_needed": rng.uniform(0, 250, n),
        "phase": rng.choice(["powerplay", "middle", "death"], n),
    }
    return pd.DataFrame(rows)[CHASE_FEATURES]


def _random_setting_frame(rng) -> pd.DataFrame:
    n = VERIFY_ROWS
    rows = {
        "score": rng.integers(0, 260, n).astype(float),
        "balls_bowled": rng.integers(0, 121, n).astype(float),
        "balls_left": rng.integers(0, 121, n).astype(float),
        "wickets_left": rng.integers(0, 11, n).astype(float),
        "current_run_rate": rng.uniform(0, 20, n),
        "innings_progress": rng.uniform(0, 1, n),
        "venue_par": rng.uniform(120, 200, n),
        "runs_last_30": rng.uniform(0, 110, n),
        "wickets_last_30": rng.integers(0, 8, n).astype(float),
        "balls_per_wicket": rng.uniform(0, 60, n),
        "phase": rng.choice(["powerplay", "middle", "death"], n),
    }
    return pd.DataFrame(rows)[SETTING_FEATURES]


def verify(code: str, chase_payload: dict, setting_arrays: dict | None) -> None:
    """Compare the exported artefacts against the estimators they came from."""
    from backend.app import inference

    rng = np.random.default_rng(RANDOM_STATE)

    frame = _random_chase_frame(rng)
    reference = joblib.load(MODEL_DIR / f"{code}_chase.joblib").predict_proba(frame)[:, 1]
    ours = inference.ChaseModel(chase_payload).predict(frame.to_dict("records"))
    worst = float(np.max(np.abs(reference - ours)))
    if worst > TOLERANCE:
        raise AssertionError(
            f"{code}: exported chase model differs from scikit-learn by {worst:.2e}"
        )
    bullet(f"{code:5} chase   matches scikit-learn to {worst:.1e} over {len(frame)} states")

    if setting_arrays is None:
        return

    frame = _random_setting_frame(rng)
    bundle = joblib.load(MODEL_DIR / f"{code}_setting.joblib")
    reference = bundle["model"].predict(frame)
    ours = inference.SettingModel(setting_arrays).predict(frame.to_dict("records"))
    worst = float(np.max(np.abs(reference - ours)))
    if worst > TOLERANCE:
        raise AssertionError(
            f"{code}: exported score model differs from scikit-learn by {worst:.2e}"
        )
    bullet(f"{code:5} setting matches scikit-learn to {worst:.1e} over {len(frame)} states")


# --------------------------------------------------------------------------

def export_league(code: str) -> bool:
    chase_payload = export_chase(code)
    if chase_payload is None:
        bullet(f"{code:5} no trained model, skipping")
        return False

    setting_arrays = export_setting(code)
    verify(code, chase_payload, setting_arrays)

    (MODEL_DIR / f"{code}_chase.json").write_text(
        json.dumps(chase_payload), encoding="utf-8"
    )
    if setting_arrays is not None:
        np.savez_compressed(MODEL_DIR / f"{code}_setting.npz", **setting_arrays)

    chase_kb = (MODEL_DIR / f"{code}_chase.json").stat().st_size / 1024
    setting_kb = (
        (MODEL_DIR / f"{code}_setting.npz").stat().st_size / 1024
        if setting_arrays is not None else 0
    )
    bullet(f"{code:5} written  chase {chase_kb:5.0f} KB   setting {setting_kb:5.0f} KB")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Export models for NumPy-only serving")
    parser.add_argument("--league", action="append", help="league code (repeatable)")
    args = parser.parse_args()

    codes = args.league or [
        lg.code for lg in LEAGUES if (MODEL_DIR / f"{lg.code}_chase.joblib").exists()
    ]

    step("Exporting models for NumPy-only inference")
    for code in codes:
        export_league(code)

    say("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
