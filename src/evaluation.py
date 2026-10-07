"""Consolidated evaluation, GRR computation, and ablation summaries.

This module is **read-only** with respect to other team members' code: it
consumes the stable result-row schema defined in
``src.baseline_models.RESULT_COLUMNS`` and ``src.baseline_models.METRIC_COLUMNS``
without duplicating any training, prediction, or alignment logic.

Result Schema (from Module 2)
-----------------------------
Every result row is a ``dict`` (or DataFrame row) with these columns:

    dataset, training_domain, test_domain, evaluation_mode, model,
    n_train, n_test, train_benign, train_attack, test_benign, test_attack,
    accuracy, precision, recall, macro_f1, mcc, far, roc_auc, pr_auc

``evaluation_mode`` is one of:
    - ``"same_dataset_holdout"``   (Module 2 baselines)
    - ``"zero_shot_cross_dataset"`` (Module 2 transfer)
    - ``"aligned_cross_dataset"``  (Module 3 after CORAL/scaling)

GRR — Generalization Recovery Rate
-----------------------------------
For each (model, transfer_direction) tuple and a given metric *m*:

    GRR(m) = (aligned_m − raw_m) / (same_dataset_m − raw_m)

*   If the denominator ``same_dataset_m − raw_m`` is **zero** the GRR is
    mathematically undefined — the raw cross-dataset score already matches
    the same-dataset ceiling so there is nothing to recover.  The function
    returns ``None`` (``NaN`` in DataFrame columns) and logs a note.
*   If the denominator is **negative** (raw > same-dataset, which is possible
    if the metric is not bounded or due to stochastic variance), the ratio is
    computed but flagged with a ``"denominator_negative"`` warning.
*   A GRR of **1.0** means full recovery; **0.0** means no improvement;
    values above 1.0 indicate the aligned model *exceeds* same-dataset
    performance on the target domain.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# ── Stable contracts re-exported for convenience ──────────────────────────
# These are the *authoritative* names defined in src.baseline_models; we
# import them at call time so this module remains importable even when the
# other members' modules are not yet merged into the working tree.

_METRIC_COLUMNS: tuple[str, ...] | None = None
_RESULT_COLUMNS: tuple[str, ...] | None = None

# Core metrics this module always reports (subset of METRIC_COLUMNS).
CORE_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "mcc",
    "far",
    "roc_auc",
)

# Evaluation modes recognised by this module.
MODE_SAME = "same_dataset_holdout"
MODE_RAW = "zero_shot_cross_dataset"
MODE_ALIGNED = "aligned_cross_dataset"

# Sentinel returned when GRR is mathematically undefined.
GRR_UNDEFINED: float | None = None  # stored as NaN in DataFrames


def _metric_columns() -> tuple[str, ...]:
    """Lazily resolve METRIC_COLUMNS from baseline_models (or use fallback)."""
    global _METRIC_COLUMNS
    if _METRIC_COLUMNS is not None:
        return _METRIC_COLUMNS
    try:
        from src.baseline_models import METRIC_COLUMNS  # type: ignore[import-untyped]
        _METRIC_COLUMNS = tuple(METRIC_COLUMNS)
    except ImportError:
        _METRIC_COLUMNS = (
            "accuracy", "precision", "recall", "macro_f1",
            "mcc", "far", "roc_auc", "pr_auc",
        )
    return _METRIC_COLUMNS


def _result_columns() -> tuple[str, ...]:
    """Lazily resolve RESULT_COLUMNS from baseline_models (or use fallback)."""
    global _RESULT_COLUMNS
    if _RESULT_COLUMNS is not None:
        return _RESULT_COLUMNS
    try:
        from src.baseline_models import RESULT_COLUMNS  # type: ignore[import-untyped]
        _RESULT_COLUMNS = tuple(RESULT_COLUMNS)
    except ImportError:
        _RESULT_COLUMNS = (
            "dataset", "training_domain", "test_domain", "evaluation_mode",
            "model", "n_train", "n_test", "train_benign", "train_attack",
            "test_benign", "test_attack",
            "accuracy", "precision", "recall", "macro_f1",
            "mcc", "far", "roc_auc", "pr_auc",
        )
    return _RESULT_COLUMNS


# ── Consolidation ─────────────────────────────────────────────────────────

def consolidate_results(
    *frames: pd.DataFrame,
) -> pd.DataFrame:
    """Stack one or more result DataFrames and validate schema consistency.

    Parameters
    ----------
    *frames:
        DataFrames matching the ``RESULT_COLUMNS`` schema from Module 2.
        Frames may originate from same-dataset baselines, raw cross-dataset
        transfer, or aligned cross-dataset runs.

    Returns
    -------
    pd.DataFrame
        A single DataFrame with consistent columns and a reset integer index.

    Raises
    ------
    ValueError
        If *frames* is empty or any frame lacks required columns.
    """
    if not frames:
        raise ValueError("consolidate_results requires at least one DataFrame")
    result_cols = set(_result_columns())
    validated: list[pd.DataFrame] = []
    for idx, frame in enumerate(frames):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Argument {idx} is not a DataFrame")
        missing = result_cols - set(frame.columns)
        if missing:
            raise ValueError(
                f"DataFrame {idx} is missing required columns: "
                f"{sorted(missing)}"
            )
        validated.append(frame.reindex(columns=list(_result_columns())))
    combined = pd.concat(validated, ignore_index=True)
    return combined


# ── GRR Calculation ───────────────────────────────────────────────────────

def compute_grr(
    same_dataset_value: float | None,
    raw_cross_value: float | None,
    aligned_cross_value: float | None,
) -> float | None:
    """Compute Generalization Recovery Rate for a single metric value.

    Parameters
    ----------
    same_dataset_value:
        Metric value from same-dataset holdout (the "ceiling").
    raw_cross_value:
        Metric value from raw (unaligned) cross-dataset transfer.
    aligned_cross_value:
        Metric value after domain alignment.

    Returns
    -------
    float or None
        The GRR ratio, or ``None`` when the denominator is zero or any
        input is missing/non-finite.

    Notes
    -----
    A ``None`` return (represented as ``NaN`` in DataFrames) means the GRR
    is **mathematically undefined** — typically because the raw score
    already equals the same-dataset ceiling, leaving zero gap to recover.
    This is *not* an error; callers should display it as "N/A" or "—".
    """
    # Guard against missing / non-finite inputs.
    for value, name in [
        (same_dataset_value, "same_dataset"),
        (raw_cross_value, "raw_cross"),
        (aligned_cross_value, "aligned_cross"),
    ]:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            LOGGER.debug("GRR undefined: %s is %r", name, value)
            return GRR_UNDEFINED

    # At this point type-checker knows values are not None.
    same = float(same_dataset_value)  # type: ignore[arg-type]
    raw = float(raw_cross_value)  # type: ignore[arg-type]
    aligned = float(aligned_cross_value)  # type: ignore[arg-type]

    denominator = same - raw
    if denominator == 0.0:
        LOGGER.info(
            "GRR undefined: same-dataset (%.4f) equals raw cross-dataset "
            "(%.4f); zero gap to recover.",
            same, raw,
        )
        return GRR_UNDEFINED

    if denominator < 0.0:
        LOGGER.warning(
            "GRR denominator is negative (same=%.4f, raw=%.4f). "
            "The raw cross-dataset score exceeds same-dataset; "
            "GRR is computed but may be misleading.",
            same, raw,
        )

    grr = (aligned - raw) / denominator
    return float(grr)


def compute_grr_table(
    consolidated: pd.DataFrame,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build a GRR summary table across all models and transfer directions.

    Parameters
    ----------
    consolidated:
        A consolidated result DataFrame containing rows with
        ``evaluation_mode`` in {``same_dataset_holdout``,
        ``zero_shot_cross_dataset``, ``aligned_cross_dataset``}.
    metrics:
        Subset of metric columns to compute GRR for.  Defaults to
        ``CORE_METRICS``.

    Returns
    -------
    pd.DataFrame
        Columns: ``model``, ``training_domain``, ``test_domain``,
        plus one ``grr_{metric}`` column for each metric.  ``NaN`` values
        denote undefined GRR (see module docstring).
    """
    if metrics is None:
        metrics = list(CORE_METRICS)
    for m in metrics:
        if m not in _metric_columns():
            raise ValueError(f"Unknown metric {m!r}")

    same = consolidated[consolidated["evaluation_mode"] == MODE_SAME]
    raw = consolidated[consolidated["evaluation_mode"] == MODE_RAW]
    aligned = consolidated[consolidated["evaluation_mode"] == MODE_ALIGNED]

    rows: list[dict] = []
    # Iterate over every (model, transfer_direction) in the raw results.
    for _, raw_row in raw.iterrows():
        model = raw_row["model"]
        train_domain = raw_row["training_domain"]
        test_domain = raw_row["test_domain"]

        # Same-dataset baseline is the model trained *and* tested on the
        # source domain.
        same_rows = same[
            (same["model"] == model)
            & (same["training_domain"] == train_domain)
            & (same["test_domain"] == train_domain)
        ]
        aligned_rows = aligned[
            (aligned["model"] == model)
            & (aligned["training_domain"] == train_domain)
            & (aligned["test_domain"] == test_domain)
        ]

        entry: dict = {
            "model": model,
            "training_domain": train_domain,
            "test_domain": test_domain,
        }
        for m in metrics:
            same_val = float(same_rows[m].iloc[0]) if len(same_rows) else None
            raw_val = float(raw_row[m]) if pd.notna(raw_row[m]) else None
            aligned_val = (
                float(aligned_rows[m].iloc[0])
                if len(aligned_rows) and pd.notna(aligned_rows[m].iloc[0])
                else None
            )
            entry[f"grr_{m}"] = compute_grr(same_val, raw_val, aligned_val)
        rows.append(entry)

    if not rows:
        cols = ["model", "training_domain", "test_domain"] + [
            f"grr_{m}" for m in metrics
        ]
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(rows)


# ── Ablation Summaries ────────────────────────────────────────────────────

def ablation_summary(
    consolidated: pd.DataFrame,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare metric means across evaluation modes for each model.

    This shows how each mode (same-dataset, raw-cross, aligned-cross)
    contributes to performance, supporting ablation-style analysis.

    Parameters
    ----------
    consolidated:
        Consolidated result DataFrame.
    metrics:
        Subset of metric columns to summarise.  Defaults to ``CORE_METRICS``.

    Returns
    -------
    pd.DataFrame
        Multi-index (model, evaluation_mode) with metric mean columns.
    """
    if metrics is None:
        metrics = list(CORE_METRICS)
    available = [m for m in metrics if m in consolidated.columns]
    if not available:
        raise ValueError("None of the requested metrics are present in the data")

    grouped = (
        consolidated
        .groupby(["model", "evaluation_mode"])[available]
        .mean()
        .reset_index()
    )
    return grouped


def metric_delta_summary(
    consolidated: pd.DataFrame,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Show raw-to-aligned metric deltas per model and direction.

    Positive deltas indicate improvement from alignment.
    """
    if metrics is None:
        metrics = list(CORE_METRICS)

    raw = consolidated[consolidated["evaluation_mode"] == MODE_RAW]
    aligned = consolidated[consolidated["evaluation_mode"] == MODE_ALIGNED]

    rows: list[dict] = []
    for _, raw_row in raw.iterrows():
        model = raw_row["model"]
        train_domain = raw_row["training_domain"]
        test_domain = raw_row["test_domain"]

        aligned_rows = aligned[
            (aligned["model"] == model)
            & (aligned["training_domain"] == train_domain)
            & (aligned["test_domain"] == test_domain)
        ]

        entry: dict = {
            "model": model,
            "training_domain": train_domain,
            "test_domain": test_domain,
        }
        for m in metrics:
            raw_val = float(raw_row[m]) if pd.notna(raw_row[m]) else None
            aligned_val = (
                float(aligned_rows[m].iloc[0])
                if len(aligned_rows) and pd.notna(aligned_rows[m].iloc[0])
                else None
            )
            if raw_val is not None and aligned_val is not None:
                entry[f"delta_{m}"] = aligned_val - raw_val
            else:
                entry[f"delta_{m}"] = None
        rows.append(entry)

    if not rows:
        cols = ["model", "training_domain", "test_domain"] + [
            f"delta_{m}" for m in metrics
        ]
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(rows)


# ── Class balance helpers ─────────────────────────────────────────────────

def class_balance_table(consolidated: pd.DataFrame) -> pd.DataFrame:
    """Extract train/test class counts per evaluation row for the dashboard."""
    cols = [
        "dataset", "evaluation_mode", "model",
        "n_train", "n_test",
        "train_benign", "train_attack",
        "test_benign", "test_attack",
    ]
    available = [c for c in cols if c in consolidated.columns]
    return consolidated[available].drop_duplicates().reset_index(drop=True)


# ── Mock / demo data generation ──────────────────────────────────────────

def generate_mock_results(seed: int = 42) -> pd.DataFrame:
    """Produce a small synthetic result table for dashboard development.

    The numbers are plausible but entirely fabricated.  Every evaluation
    mode is represented so the dashboard can render all views.

    The mock data uses two transfer directions:
      - CICIDS-2017 → UNSW-NB15
      - UNSW-NB15 → CICIDS-2017
    and three models (decision_tree, random_forest, logistic_regression).
    """
    rng = np.random.default_rng(seed)
    models = ["decision_tree", "random_forest", "logistic_regression"]
    datasets = [
        ("cicids2017", "unsw_nb15"),
        ("unsw_nb15", "cicids2017"),
    ]

    rows: list[dict] = []

    def _metrics(
        acc_base: float, noise: float = 0.03
    ) -> dict[str, float | None]:
        acc = min(max(acc_base + rng.normal(0, noise), 0.0), 1.0)
        prec = min(max(acc + rng.normal(0, noise), 0.0), 1.0)
        rec = min(max(acc + rng.normal(-0.02, noise), 0.0), 1.0)
        f1 = min(max(2 * prec * rec / (prec + rec + 1e-9), 0.0), 1.0)
        mcc = min(max(acc * 2 - 1 + rng.normal(0, noise), -1.0), 1.0)
        far = min(max(1 - acc + rng.normal(0, noise / 2), 0.0), 1.0)
        auc = min(max(acc + rng.normal(0.02, noise), 0.0), 1.0)
        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "macro_f1": round(f1, 4),
            "mcc": round(mcc, 4),
            "far": round(far, 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(auc - 0.01, 4),
        }

    # Same-dataset baselines for each domain.
    for domain in ("cicids2017", "unsw_nb15"):
        for model in models:
            m = _metrics(0.92 + rng.uniform(-0.03, 0.03))
            rows.append({
                "dataset": domain,
                "training_domain": domain,
                "test_domain": domain,
                "evaluation_mode": MODE_SAME,
                "model": model,
                "n_train": 800,
                "n_test": 200,
                "train_benign": 400,
                "train_attack": 400,
                "test_benign": 100,
                "test_attack": 100,
                **m,
            })

    # Cross-dataset: raw and aligned.
    for src, tgt in datasets:
        for model in models:
            raw_m = _metrics(0.60 + rng.uniform(-0.05, 0.05))
            aligned_m = _metrics(0.80 + rng.uniform(-0.03, 0.03))
            for mode, metrics in [
                (MODE_RAW, raw_m),
                (MODE_ALIGNED, aligned_m),
            ]:
                rows.append({
                    "dataset": f"{src}_to_{tgt}",
                    "training_domain": src,
                    "test_domain": tgt,
                    "evaluation_mode": mode,
                    "model": model,
                    "n_train": 1000,
                    "n_test": 1000,
                    "train_benign": 500,
                    "train_attack": 500,
                    "test_benign": 500,
                    "test_attack": 500,
                    **metrics,
                })

    return pd.DataFrame(rows, columns=list(_result_columns()))
