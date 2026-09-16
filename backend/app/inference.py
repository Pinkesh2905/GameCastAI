"""
Model evaluation in NumPy alone.

This is the serving-time replacement for scikit-learn. It reads the artefacts
`backend.ml.export_models` writes and reproduces the fitted estimators exactly
- `export_models` asserts agreement to 1e-9 before it will write anything, so
these two files have to be changed together.

Keeping the runtime to NumPy takes the deployed dependency set from about
415 MB to about 71 MB, which is the difference between fitting in a serverless
function and not. It also removes roughly two seconds of import time from a
cold start, which matters more than it sounds like on a platform that goes
cold between visitors.
"""

from __future__ import annotations

import json

import numpy as np


def _as_matrix(rows: list, names: list) -> np.ndarray:
    """Stack a list of feature dicts into a float matrix in a fixed order."""
    return np.array(
        [[float(row[name]) for name in names] for row in rows],
        dtype=np.float64,
    )


class ChaseModel:
    """Logistic regression, optionally averaged over isotonic-calibrated folds.

    The pipeline being reproduced is StandardScaler on the numeric columns,
    OneHotEncoder on the categorical ones, then a linear model. That is:

        z = ((x - mean) / scale) . w_num + onehot . w_cat + b
        p = 1 / (1 + exp(-z))

    A calibrated model repeats that for each cross-validation fold, maps each
    result through that fold's isotonic step function, and averages.
    """

    def __init__(self, payload: dict):
        self.numeric = list(payload["numeric"])
        self.categorical = list(payload["categorical"])
        self.calibrated = payload["kind"] == "calibrated_logistic"

        self.members = []
        for member in payload["members"]:
            entry = {
                "mean": np.asarray(member["scaler_mean"], dtype=np.float64),
                "scale": np.asarray(member["scaler_scale"], dtype=np.float64),
                "categories": [list(c) for c in member["categories"]],
                "coef": np.asarray(member["coef"], dtype=np.float64),
                "intercept": float(member["intercept"]),
            }
            if "isotonic" in member:
                entry["iso_x"] = np.asarray(member["isotonic"]["x"], dtype=np.float64)
                entry["iso_y"] = np.asarray(member["isotonic"]["y"], dtype=np.float64)
            self.members.append(entry)

    def _one_hot(self, rows: list, categories: list) -> np.ndarray:
        """Reproduce OneHotEncoder(handle_unknown="ignore").

        An unseen level yields an all-zero block rather than an error, which
        is what "ignore" means and why a new venue or a new phase label cannot
        take the service down.
        """
        blocks = []
        for index, name in enumerate(self.categorical):
            levels = categories[index]
            lookup = {level: position for position, level in enumerate(levels)}
            block = np.zeros((len(rows), len(levels)), dtype=np.float64)
            for row_index, row in enumerate(rows):
                position = lookup.get(str(row[name]))
                if position is not None:
                    block[row_index, position] = 1.0
            blocks.append(block)
        return np.hstack(blocks) if blocks else np.zeros((len(rows), 0))

    @staticmethod
    def _sigmoid(logit: np.ndarray) -> np.ndarray:
        """Evaluated in the stable direction for each sign, so a one-sided
        chase producing a large negative logit cannot overflow the exponential."""
        clipped = np.clip(logit, -700, 700)
        positive = 1.0 / (1.0 + np.exp(-clipped))
        exponential = np.exp(clipped)
        negative = exponential / (1.0 + exponential)
        return np.where(clipped >= 0, positive, negative)

    @staticmethod
    def _isotonic(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """IsotonicRegression(out_of_bounds="clip") as a lookup.

        scikit-learn interpolates linearly between thresholds and clamps
        outside them, which is exactly what np.interp does at the edges.
        """
        if len(x) == 0:
            return values
        return np.interp(values, x, y, left=y[0], right=y[-1])

    def predict(self, rows: list) -> np.ndarray:
        """Win probability for each feature dict."""
        if not rows:
            return np.zeros(0)

        numeric = _as_matrix(rows, self.numeric)
        totals = np.zeros(len(rows), dtype=np.float64)

        for member in self.members:
            scaled = (numeric - member["mean"]) / member["scale"]
            encoded = self._one_hot(rows, member["categories"])
            design = np.hstack([scaled, encoded])

            logit = design @ member["coef"] + member["intercept"]

            if "iso_x" in member:
                # CalibratedClassifierCV prefers decision_function over
                # predict_proba, so the isotonic map was fitted on the RAW
                # LOGIT, not on a probability. Feeding it a sigmoid output
                # instead looks plausible and is wrong by up to 0.47.
                probability = self._isotonic(logit, member["iso_x"], member["iso_y"])
            else:
                probability = self._sigmoid(logit)

            totals += probability

        return totals / len(self.members)


class SettingModel:
    """Gradient-boosted regression trees, walked node by node.

    Every tree is stored in one flat table with an offset per tree, so the
    whole ensemble is a handful of contiguous arrays rather than 150 objects.
    """

    def __init__(self, arrays):
        self.value = np.asarray(arrays["value"])
        self.feature_idx = np.asarray(arrays["feature_idx"])
        self.num_threshold = np.asarray(arrays["num_threshold"])
        self.missing_go_to_left = np.asarray(arrays["missing_go_to_left"])
        self.left = np.asarray(arrays["left"])
        self.right = np.asarray(arrays["right"])
        self.is_leaf = np.asarray(arrays["is_leaf"])
        self.is_categorical = np.asarray(arrays["is_categorical"])
        self.bitset_idx = np.asarray(arrays["bitset_idx"])
        self.tree_offset = np.asarray(arrays["tree_offset"])
        self.bitsets = np.asarray(arrays["bitsets"])
        self.baseline = float(np.asarray(arrays["baseline"]).ravel()[0])

        # The column order the trees actually index into. It is NOT the order
        # the features were listed in at training time: HistGradientBoosting
        # hoists categorical columns to the front internally, so the exporter
        # reads the real order off the fitted estimator and stores it here.
        self.tree_order = [str(x) for x in np.asarray(arrays["tree_order"])]
        labels = [str(x) for x in np.asarray(arrays["phase_labels"])]
        values = np.asarray(arrays["phase_values"], dtype=np.float64)
        self.phase_values = dict(zip(labels, values))
        self.offsets = json.loads(str(np.asarray(arrays["offsets_json"]).ravel()[0]))

    def _design(self, rows: list) -> np.ndarray:
        """Build the matrix in the order the trees expect.

        A label the model never saw becomes NaN and takes the missing-value
        branch, which is what scikit-learn does with an unknown category.
        """
        matrix = np.empty((len(rows), len(self.tree_order)), dtype=np.float64)
        for column, name in enumerate(self.tree_order):
            if name == "phase":
                matrix[:, column] = [
                    self.phase_values.get(str(row[name]), np.nan) for row in rows
                ]
            else:
                matrix[:, column] = [float(row[name]) for row in rows]
        return matrix

    def _in_bitset(self, bitset_index: int, category: float) -> bool:
        """Whether a category is in a node's left-hand set.

        scikit-learn packs the set of categories that go left into eight
        uint32s, so membership is a bit test. Anything outside 0..255 is
        treated as missing, matching the library.
        """
        if not np.isfinite(category):
            return False
        value = int(category)
        if value < 0 or value >= 256 or bitset_index < 0:
            return False
        word = self.bitsets[bitset_index][value // 32]
        return bool((int(word) >> (value % 32)) & 1)

    def predict(self, rows: list) -> np.ndarray:
        if not rows:
            return np.zeros(0)

        design = self._design(rows)
        totals = np.full(len(rows), self.baseline, dtype=np.float64)

        n_trees = len(self.tree_offset) - 1
        for row_index in range(len(rows)):
            sample = design[row_index]
            total = 0.0
            for tree in range(n_trees):
                node = int(self.tree_offset[tree])
                while not self.is_leaf[node]:
                    feature = sample[self.feature_idx[node]]
                    if np.isnan(feature):
                        go_left = bool(self.missing_go_to_left[node])
                    elif self.is_categorical[node]:
                        go_left = self._in_bitset(int(self.bitset_idx[node]), feature)
                        if not go_left and (feature < 0 or feature >= 256):
                            go_left = bool(self.missing_go_to_left[node])
                    else:
                        go_left = feature <= self.num_threshold[node]
                    base = int(self.tree_offset[tree])
                    node = base + int(self.left[node] if go_left else self.right[node])
                total += self.value[node]
            totals[row_index] += total

        return totals

    def band(self, balls_bowled: int) -> dict:
        """The conformal offsets for this stage of the innings."""
        bucket = min(int(balls_bowled) // 30, 3)
        by_bucket = self.offsets.get("by_bucket", {})
        return (
            by_bucket.get(str(bucket))
            or by_bucket.get(bucket)
            or self.offsets.get("fallback", {"low": -25.0, "high": 25.0})
        )
