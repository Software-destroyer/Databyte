#!/usr/bin/env python
"""CLI orchestration for the cross-dataset intrusion detection pipeline.

Usage
-----
::

    # Show help
    python main.py --help

    # Run mock / small workflow (no real datasets required)
    python main.py mock

    # Generate mock CSV fixtures
    python main.py generate-mock [--rows 500] [--output-dir data/raw]

    # Full pipeline steps (require real or preprocessed data)
    python main.py preprocess --cicids <csv> --unsw <csv>
    python main.py baselines  [--models decision_tree random_forest]
    python main.py transfer   [--models decision_tree random_forest]
    python main.py evaluate   [--results-csv results/consolidated_results.csv]
    python main.py dashboard

    # Run the full pipeline end-to-end (preprocess → baselines → transfer → evaluate)
    python main.py run --cicids <csv> --unsw <csv>

Design notes
~~~~~~~~~~~~~
*   This module **never** duplicates logic from Modules 1-3.  It calls their
    public APIs when available and fails with descriptive errors when the
    required modules are not yet merged.
*   The ``mock`` command works without any dependencies beyond Module 4
    (``src.evaluation``).
*   The ``dashboard`` command simply launches Streamlit as a subprocess.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence

# Ensure project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("main")


# ── Helpers ───────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _check_module(module_name: str, friendly_name: str) -> None:
    """Import-check a peer module and give a clear error if absent."""
    try:
        __import__(module_name)
    except ImportError as exc:
        LOGGER.error(
            "Module '%s' (%s) is not available.  "
            "Ensure the corresponding team member's branch has been merged "
            "or the module is on sys.path.  Original error: %s",
            module_name, friendly_name, exc,
        )
        raise SystemExit(1) from exc


def _check_file(path: Path, description: str) -> None:
    """Abort with a clear message if a required file is missing."""
    if not path.is_file():
        LOGGER.error(
            "%s not found at %s.  "
            "Run the prerequisite pipeline step first, or use 'python main.py mock' "
            "to work with synthetic data.",
            description, path,
        )
        raise SystemExit(1)


# ── Command: mock ─────────────────────────────────────────────────────────

def cmd_mock(args: argparse.Namespace) -> None:
    """Generate mock evaluation results and display a summary."""
    from src.evaluation import (
        ablation_summary,
        compute_grr_table,
        consolidate_results,
        generate_mock_results,
    )

    LOGGER.info("Generating mock evaluation results...")
    mock = generate_mock_results(seed=args.seed)
    consolidated = consolidate_results(mock)
    # Keep synthetic demonstration rows visibly marked so they cannot be
    # mistaken for experimental results in the dashboard or a report.
    consolidated["data_source"] = "synthetic_demo"

    output_dir = _ensure_dir(RESULTS_DIR)
    output_path = output_dir / "consolidated_results.csv"
    consolidated.to_csv(output_path, index=False)
    LOGGER.info("Wrote %d rows to %s", len(consolidated), output_path)

    # Print summary tables.
    print("\n=== Ablation Summary ===")
    abl = ablation_summary(consolidated)
    print(abl.to_string(index=False))

    print("\n=== GRR Table ===")
    grr = compute_grr_table(consolidated)
    print(grr.to_string(index=False))

    print(f"\nResults saved to: {output_path}")
    print("Launch the dashboard with:  python main.py dashboard")


# -- Command: generate-mock ------------------------------------------------

def cmd_generate_mock(args: argparse.Namespace) -> None:
    """Generate small mock CSV fixtures for testing."""
    from tests.mock_data_generator import generate_mock_data

    paths = generate_mock_data(
        output_dir=args.output_dir,
        rows=args.rows,
        seed=args.seed,
    )
    for dataset, path in paths.items():
        LOGGER.info("Generated %s: %s", dataset, path)


# -- Command: preprocess --------------------------------------------------

def cmd_preprocess(args: argparse.Namespace) -> None:
    """Run Module 1 preprocessing on real CSV files."""
    _check_module("src.preprocessing", "Module 1 - Preprocessing")
    from src.preprocessing import DataPreprocessor

    for raw_path, label in ((Path(args.cicids), "CICIDS-2017 CSV path"),
                            (Path(args.unsw), "UNSW-NB15 CSV path")):
        if not raw_path.exists():
            LOGGER.error("%s not found: %s", label, raw_path)
            raise SystemExit(1)

    processor = DataPreprocessor(output_dir=args.output_dir)
    paths = processor.process_datasets(args.cicids, args.unsw)
    from src.feature_mapping import FeatureMapper
    mapper = FeatureMapper(output_dir=args.output_dir)
    for dataset, path in paths.items():
        aligned = mapper.process_file(path, dataset=dataset)
        LOGGER.info("Preprocessed %s -> %s; canonical features -> %s", dataset, path, aligned)


# -- Command: baselines ---------------------------------------------------

def cmd_baselines(args: argparse.Namespace) -> None:
    """Run Module 2 same-dataset baseline evaluation."""
    _check_module("src.baseline_models", "Module 2 - Baselines")
    from src.baseline_models import evaluate_dataset

    import pandas as pd

    processed_dir = Path(getattr(args, "output_dir", getattr(args, "processed_dir", PROCESSED_DATA_DIR)))
    cic_path = processed_dir / "cic_aligned.parquet"
    unsw_path = processed_dir / "unsw_aligned.parquet"
    _check_file(cic_path, "Canonical CICIDS-2017 Parquet")
    _check_file(unsw_path, "Canonical UNSW-NB15 Parquet")

    cic_df = pd.read_parquet(cic_path)
    unsw_df = pd.read_parquet(unsw_path)

    models = args.models if args.models else None
    cic_results = evaluate_dataset("cicids2017", cic_df, models=models)
    unsw_results = evaluate_dataset("unsw_nb15", unsw_df, models=models)

    combined = pd.concat([cic_results, unsw_results], ignore_index=True)
    output = _ensure_dir(RESULTS_DIR) / "baseline_results.csv"
    combined.to_csv(output, index=False)
    LOGGER.info("Baseline results (%d rows) -> %s", len(combined), output)
    print(combined.to_string(index=False))


# -- Command: transfer ----------------------------------------------------

def cmd_transfer(args: argparse.Namespace) -> None:
    """Run Module 2 bidirectional cross-dataset transfer."""
    _check_module("src.cross_dataset", "Module 2 - Cross-dataset")
    from src.cross_dataset import evaluate_bidirectional

    import pandas as pd

    processed_dir = Path(getattr(args, "output_dir", getattr(args, "processed_dir", PROCESSED_DATA_DIR)))
    cic_path = processed_dir / "cic_aligned.parquet"
    unsw_path = processed_dir / "unsw_aligned.parquet"
    _check_file(cic_path, "Canonical CICIDS-2017 Parquet")
    _check_file(unsw_path, "Canonical UNSW-NB15 Parquet")

    cic_df = pd.read_parquet(cic_path)
    unsw_df = pd.read_parquet(unsw_path)

    models = args.models if args.models else None
    results = evaluate_bidirectional(cic_df, unsw_df, models=models)

    output = _ensure_dir(RESULTS_DIR) / "cross_dataset_results.csv"
    results.to_csv(output, index=False)
    LOGGER.info("Transfer results (%d rows) -> %s", len(results), output)
    print(results.to_string(index=False))


def cmd_align(args: argparse.Namespace) -> None:
    """Run divergence analysis and CORAL transfer in both directions."""
    import pandas as pd
    from src.domain_alignment import align_bidirectional
    from src.evaluation import consolidate_results

    processed_dir = Path(getattr(args, "output_dir", getattr(args, "processed_dir", PROCESSED_DATA_DIR)))
    cic_path = processed_dir / "cic_aligned.parquet"
    unsw_path = processed_dir / "unsw_aligned.parquet"
    _check_file(cic_path, "Canonical CICIDS-2017 Parquet")
    _check_file(unsw_path, "Canonical UNSW-NB15 Parquet")
    cic_df, unsw_df = pd.read_parquet(cic_path), pd.read_parquet(unsw_path)
    aligned, details = align_bidirectional(cic_df, unsw_df, models=args.models,
                                           lambda_reg=args.lambda_reg)
    _ensure_dir(RESULTS_DIR)
    aligned.to_csv(RESULTS_DIR / "aligned_results.csv", index=False)
    report_rows = []
    for direction, detail in details.items():
        for phase in ("divergence_before", "divergence_after"):
            report = detail[phase]
            w = report["wasserstein"].set_index("feature")["wasserstein_dist"]
            k = report["kl"].set_index("feature")["kl_divergence"]
            for feature in w.index.union(k.index):
                report_rows.append({"direction": direction, "phase": phase,
                    "feature": feature, "wasserstein_distance": w.get(feature),
                    "kl_divergence": k.get(feature), "mmd_squared": report["mmd"]})
    pd.DataFrame(report_rows).to_csv(RESULTS_DIR / "divergence_report.csv", index=False)
    LOGGER.info("Aligned results (%d rows) and divergence report saved to %s", len(aligned), RESULTS_DIR)
    print(aligned.to_string(index=False))


# -- Command: evaluate ----------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> None:
    """Consolidate results and compute GRR / ablation reports."""
    import pandas as pd
    from src.evaluation import (
        ablation_summary,
        compute_grr_table,
        consolidate_results,
        metric_delta_summary,
    )

    # Collect available result files.
    result_files: list[Path] = []
    if args.results_csv:
        path = Path(args.results_csv)
        _check_file(path, "Specified results CSV")
        result_files.append(path)
    else:
        for name in [
            "baseline_results.csv",
            "cross_dataset_results.csv",
            "aligned_results.csv",
        ]:
            candidate = RESULTS_DIR / name
            if candidate.is_file():
                result_files.append(candidate)
                LOGGER.info("Found %s", candidate)

    if not result_files:
        LOGGER.error(
            "No result files found in %s.  "
            "Run baselines/transfer/alignment first, or pass --results-csv.",
            RESULTS_DIR,
        )
        raise SystemExit(1)

    frames = [pd.read_csv(f) for f in result_files]
    consolidated = consolidate_results(*frames)

    output = _ensure_dir(RESULTS_DIR) / "consolidated_results.csv"
    consolidated.to_csv(output, index=False)
    LOGGER.info("Consolidated %d rows -> %s", len(consolidated), output)

    print("\n=== Ablation Summary ===")
    abl = ablation_summary(consolidated)
    print(abl.to_string(index=False))

    print("\n=== GRR Table ===")
    grr = compute_grr_table(consolidated)
    if len(grr):
        print(grr.to_string(index=False))
    else:
        print("  (Not enough data to compute GRR - need same-dataset, raw, "
              "and aligned results.)")

    print("\n=== Metric Deltas (Aligned - Raw) ===")
    deltas = metric_delta_summary(consolidated)
    if len(deltas):
        print(deltas.to_string(index=False))
    else:
        print("  (No raw+aligned pairs available.)")

    # Also write machine-readable reports.
    grr_path = _ensure_dir(RESULTS_DIR) / "grr_report.csv"
    grr.to_csv(grr_path, index=False)
    LOGGER.info("GRR report -> %s", grr_path)


# -- Command: dashboard ---------------------------------------------------

def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit dashboard."""
    app_path = PROJECT_ROOT / "dashboard" / "app.py"
    if not app_path.is_file():
        LOGGER.error("Dashboard app not found at %s", app_path)
        raise SystemExit(1)

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if args.port:
        cmd += ["--server.port", str(args.port)]
    LOGGER.info("Launching: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        LOGGER.info("Dashboard stopped.")
    except FileNotFoundError:
        LOGGER.error(
            "Streamlit is not installed.  Run: pip install streamlit"
        )
        raise SystemExit(1)


# -- Command: run (full pipeline) -----------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Run the full pipeline: preprocess -> baselines -> transfer -> evaluate."""
    LOGGER.info("=== Full Pipeline ===")

    # Step 1: Preprocess.
    LOGGER.info("Step 1/5: Preprocessing and canonical mapping...")
    cmd_preprocess(args)

    # Step 2: Baselines.
    LOGGER.info("Step 2/5: Same-dataset baselines...")
    cmd_baselines(args)

    # Step 3: Cross-dataset transfer.
    LOGGER.info("Step 3/5: Cross-dataset transfer...")
    cmd_transfer(args)

    # Step 4: Unsupervised alignment and aligned cross-domain evaluation.
    LOGGER.info("Step 4/5: Divergence analysis and CORAL alignment...")
    cmd_align(args)

    # Step 5: Consolidate and report retention/recovery metrics.
    LOGGER.info("Step 5/5: Consolidating evaluation results...")
    args.results_csv = None
    cmd_evaluate(args)
    LOGGER.info("Pipeline complete. Launch the dashboard with: python main.py dashboard")


# ── Argument parser ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Cross-Dataset Intrusion Detection — CLI Orchestration\n\n"
            "Module 4: consolidated evaluation, GRR/ablation reporting, "
            "Streamlit dashboard, and CLI orchestration."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    # mock
    sp_mock = subparsers.add_parser(
        "mock",
        help="Generate mock results and display a summary (no real data needed)",
    )
    sp_mock.add_argument("--seed", type=int, default=42)
    sp_mock.set_defaults(func=cmd_mock)

    # generate-mock
    sp_genmock = subparsers.add_parser(
        "generate-mock",
        help="Generate small mock CSV fixtures",
    )
    sp_genmock.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    sp_genmock.add_argument("--rows", type=int, default=1000)
    sp_genmock.add_argument("--seed", type=int, default=42)
    sp_genmock.set_defaults(func=cmd_generate_mock)

    # preprocess
    sp_pre = subparsers.add_parser(
        "preprocess",
        help="Run preprocessing on raw CSV files (Module 1)",
    )
    sp_pre.add_argument("--cicids", required=True, help="Path to CICIDS-2017 CSV")
    sp_pre.add_argument("--unsw", required=True, help="Path to UNSW-NB15 CSV")
    sp_pre.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    sp_pre.set_defaults(func=cmd_preprocess)

    # baselines
    sp_base = subparsers.add_parser(
        "baselines",
        help="Run same-dataset baseline evaluation (Module 2)",
    )
    sp_base.add_argument("--models", nargs="+", default=None)
    sp_base.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    sp_base.set_defaults(func=cmd_baselines)

    # transfer
    sp_xfer = subparsers.add_parser(
        "transfer",
        help="Run bidirectional cross-dataset transfer (Module 2)",
    )
    sp_xfer.add_argument("--models", nargs="+", default=None)
    sp_xfer.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    sp_xfer.set_defaults(func=cmd_transfer)

    # align
    sp_align = subparsers.add_parser(
        "align", help="Run divergence analysis and CORAL aligned transfer (Module 3)"
    )
    sp_align.add_argument("--models", nargs="+", default=None)
    sp_align.add_argument("--lambda-reg", type=float, default=1e-3)
    sp_align.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    sp_align.set_defaults(func=cmd_align)

    # evaluate
    sp_eval = subparsers.add_parser(
        "evaluate",
        help="Consolidate results and compute GRR / ablation reports",
    )
    sp_eval.add_argument(
        "--results-csv",
        default=None,
        help="Path to a pre-consolidated CSV (skips auto-discovery)",
    )
    sp_eval.set_defaults(func=cmd_evaluate)

    # dashboard
    sp_dash = subparsers.add_parser(
        "dashboard",
        help="Launch the Streamlit dashboard",
    )
    sp_dash.add_argument("--port", type=int, default=None)
    sp_dash.set_defaults(func=cmd_dashboard)

    # run (full pipeline)
    sp_run = subparsers.add_parser(
        "run",
        help="Run the full pipeline: preprocess → baselines → transfer → evaluate",
    )
    sp_run.add_argument("--cicids", required=True, help="Path to CICIDS-2017 CSV")
    sp_run.add_argument("--unsw", required=True, help="Path to UNSW-NB15 CSV")
    sp_run.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    sp_run.add_argument("--models", nargs="+", default=None)
    sp_run.add_argument("--results-csv", default=None)
    sp_run.add_argument("--lambda-reg", type=float, default=1e-3)
    sp_run.set_defaults(func=cmd_run)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    args.func(args)


if __name__ == "__main__":
    main()
