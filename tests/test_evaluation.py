"""Tests for src.evaluation — consolidation, GRR, and ablation logic."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    CORE_METRICS,
    GRR_UNDEFINED,
    MODE_ALIGNED,
    MODE_RAW,
    MODE_SAME,
    ablation_summary,
    class_balance_table,
    compute_grr,
    compute_grr_table,
    consolidate_results,
    generate_mock_results,
    metric_delta_summary,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_row(
    model: str = "decision_tree",
    mode: str = MODE_SAME,
    train_domain: str = "cicids2017",
    test_domain: str = "cicids2017",
    accuracy: float = 0.90,
    **metric_overrides,
) -> dict:
    """Build a single result-schema row with sensible defaults."""
    base_metrics = {
        "accuracy": accuracy,
        "precision": accuracy + 0.01,
        "recall": accuracy - 0.01,
        "macro_f1": accuracy,
        "mcc": accuracy * 2 - 1,
        "far": 1 - accuracy,
        "roc_auc": accuracy + 0.02,
        "pr_auc": accuracy + 0.01,
    }
    base_metrics.update(metric_overrides)
    ds = train_domain if mode == MODE_SAME else f"{train_domain}_to_{test_domain}"
    return {
        "dataset": ds,
        "training_domain": train_domain,
        "test_domain": test_domain,
        "evaluation_mode": mode,
        "model": model,
        "n_train": 800,
        "n_test": 200,
        "train_benign": 400,
        "train_attack": 400,
        "test_benign": 100,
        "test_attack": 100,
        **base_metrics,
    }


def _build_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ── consolidate_results ──────────────────────────────────────────────────

class TestConsolidateResults:
    def test_single_frame(self):
        df = _build_df(_make_row())
        result = consolidate_results(df)
        assert len(result) == 1
        assert "accuracy" in result.columns

    def test_multiple_frames(self):
        df1 = _build_df(_make_row(model="dt"))
        df2 = _build_df(_make_row(model="rf"))
        result = consolidate_results(df1, df2)
        assert len(result) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            consolidate_results()

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"model": ["dt"], "accuracy": [0.9]})
        with pytest.raises(ValueError, match="missing required columns"):
            consolidate_results(df)

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="not a DataFrame"):
            consolidate_results("not a dataframe")  # type: ignore[arg-type]


# ── compute_grr ──────────────────────────────────────────────────────────

class TestComputeGRR:
    def test_perfect_recovery(self):
        # same=0.90, raw=0.60, aligned=0.90 → GRR=1.0
        assert compute_grr(0.90, 0.60, 0.90) == pytest.approx(1.0)

    def test_no_recovery(self):
        # same=0.90, raw=0.60, aligned=0.60 → GRR=0.0
        assert compute_grr(0.90, 0.60, 0.60) == pytest.approx(0.0)

    def test_partial_recovery(self):
        # same=0.90, raw=0.60, aligned=0.75 → GRR=0.5
        assert compute_grr(0.90, 0.60, 0.75) == pytest.approx(0.5)

    def test_exceeds_same_dataset(self):
        # same=0.90, raw=0.60, aligned=0.95 → GRR > 1.0
        grr = compute_grr(0.90, 0.60, 0.95)
        assert grr is not None
        assert grr > 1.0

    def test_zero_denominator_returns_undefined(self):
        # same=0.60, raw=0.60 → denominator is zero
        result = compute_grr(0.60, 0.60, 0.80)
        assert result is GRR_UNDEFINED
        assert result is None

    def test_none_input_returns_undefined(self):
        assert compute_grr(None, 0.60, 0.80) is GRR_UNDEFINED
        assert compute_grr(0.90, None, 0.80) is GRR_UNDEFINED
        assert compute_grr(0.90, 0.60, None) is GRR_UNDEFINED

    def test_nan_input_returns_undefined(self):
        assert compute_grr(float("nan"), 0.60, 0.80) is GRR_UNDEFINED

    def test_inf_input_returns_undefined(self):
        assert compute_grr(float("inf"), 0.60, 0.80) is GRR_UNDEFINED

    def test_negative_denominator_still_computes(self):
        # raw > same → negative denominator, but we still compute
        grr = compute_grr(0.50, 0.70, 0.80)
        assert grr is not None
        assert isinstance(grr, float)


# ── compute_grr_table ────────────────────────────────────────────────────

class TestComputeGRRTable:
    def test_basic_grr_table(self):
        rows = [
            _make_row(model="dt", mode=MODE_SAME, accuracy=0.90),
            _make_row(model="dt", mode=MODE_RAW,
                      train_domain="cicids2017", test_domain="unsw_nb15",
                      accuracy=0.60),
            _make_row(model="dt", mode=MODE_ALIGNED,
                      train_domain="cicids2017", test_domain="unsw_nb15",
                      accuracy=0.80),
        ]
        df = _build_df(*rows)
        grr = compute_grr_table(df, metrics=["accuracy"])
        assert len(grr) == 1
        assert "grr_accuracy" in grr.columns
        # (0.80 - 0.60) / (0.90 - 0.60) = 0.667
        assert grr["grr_accuracy"].iloc[0] == pytest.approx(2 / 3, abs=1e-4)

    def test_empty_when_no_raw(self):
        df = _build_df(_make_row(model="dt", mode=MODE_SAME))
        grr = compute_grr_table(df)
        assert len(grr) == 0

    def test_undefined_when_no_aligned(self):
        rows = [
            _make_row(model="dt", mode=MODE_SAME, accuracy=0.90),
            _make_row(model="dt", mode=MODE_RAW,
                      train_domain="cicids2017", test_domain="unsw_nb15",
                      accuracy=0.60),
        ]
        df = _build_df(*rows)
        grr = compute_grr_table(df, metrics=["accuracy"])
        assert len(grr) == 1
        assert pd.isna(grr["grr_accuracy"].iloc[0])

    def test_unknown_metric_raises(self):
        df = _build_df(_make_row())
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_grr_table(df, metrics=["nonexistent_metric"])


# ── ablation_summary ─────────────────────────────────────────────────────

class TestAblationSummary:
    def test_groups_by_model_and_mode(self):
        rows = [
            _make_row(model="dt", mode=MODE_SAME, accuracy=0.90),
            _make_row(model="dt", mode=MODE_RAW,
                      train_domain="cic", test_domain="unsw",
                      accuracy=0.60),
            _make_row(model="rf", mode=MODE_SAME, accuracy=0.92),
        ]
        df = _build_df(*rows)
        abl = ablation_summary(df, metrics=["accuracy"])
        assert "model" in abl.columns
        assert "evaluation_mode" in abl.columns
        assert len(abl) == 3

    def test_no_matching_metrics_raises(self):
        df = _build_df(_make_row())
        with pytest.raises(ValueError, match="None of the requested metrics"):
            ablation_summary(df, metrics=["nonexistent"])


# ── metric_delta_summary ─────────────────────────────────────────────────

class TestMetricDeltaSummary:
    def test_positive_delta(self):
        rows = [
            _make_row(model="dt", mode=MODE_RAW,
                      train_domain="cic", test_domain="unsw",
                      accuracy=0.60),
            _make_row(model="dt", mode=MODE_ALIGNED,
                      train_domain="cic", test_domain="unsw",
                      accuracy=0.80),
        ]
        df = _build_df(*rows)
        deltas = metric_delta_summary(df, metrics=["accuracy"])
        assert len(deltas) == 1
        assert deltas["delta_accuracy"].iloc[0] == pytest.approx(0.20)

    def test_missing_aligned_gives_none_delta(self):
        rows = [
            _make_row(model="dt", mode=MODE_RAW,
                      train_domain="cic", test_domain="unsw",
                      accuracy=0.60),
        ]
        df = _build_df(*rows)
        deltas = metric_delta_summary(df, metrics=["accuracy"])
        assert len(deltas) == 1
        assert deltas["delta_accuracy"].iloc[0] is None


# ── class_balance_table ──────────────────────────────────────────────────

class TestClassBalanceTable:
    def test_extracts_counts(self):
        df = _build_df(_make_row())
        balance = class_balance_table(df)
        assert "train_benign" in balance.columns
        assert len(balance) == 1


# ── generate_mock_results ────────────────────────────────────────────────

class TestGenerateMockResults:
    def test_produces_all_modes(self):
        mock = generate_mock_results()
        modes = set(mock["evaluation_mode"].unique())
        assert MODE_SAME in modes
        assert MODE_RAW in modes
        assert MODE_ALIGNED in modes

    def test_deterministic(self):
        a = generate_mock_results(seed=123)
        b = generate_mock_results(seed=123)
        pd.testing.assert_frame_equal(a, b)

    def test_all_core_metrics_present(self):
        mock = generate_mock_results()
        for m in CORE_METRICS:
            assert m in mock.columns

    def test_reasonable_value_ranges(self):
        mock = generate_mock_results()
        for m in ["accuracy", "precision", "recall", "macro_f1"]:
            vals = mock[m].dropna()
            assert (vals >= 0).all() and (vals <= 1).all(), f"{m} out of range"
