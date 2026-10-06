"""Generate small, deterministic CSV fixtures for both benchmark schemas."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def generate_mock_data(
    output_dir: str | Path = DEFAULT_RAW_DIR,
    rows: int = 1_000,
    seed: int = 42,
) -> dict[str, Path]:
    """Write deterministic mock CSVs to dataset-specific raw directories.

    CICIDS mock columns match the documented CICFlowMeter names. UNSW columns
    match the documented UNSW-NB15 feature and label names, including
    ``attack_cat`` and the binary ``label`` field.
    """
    if rows < 4:
        raise ValueError("rows must be at least 4 to include both labels")
    destination = Path(output_dir)
    cic_dir = destination / "cicids2017"
    unsw_dir = destination / "unsw_nb15"
    cic_dir.mkdir(parents=True, exist_ok=True)
    unsw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    attacks = rng.random(rows) < 0.2
    # Include rate infinities, NaNs, and a constant field to exercise cleanup.
    cic_rate = rng.lognormal(mean=3.0, sigma=0.8, size=rows)
    cic_rate[0] = np.inf
    cic_rate[1] = -np.inf
    cic_rate[2] = np.nan
    cic = pd.DataFrame(
        {
            "Flow Duration": rng.integers(1, 2_000_000, size=rows),
            "Total Fwd Packets": rng.integers(1, 100, size=rows),
            "Total Backward Packets": rng.integers(1, 90, size=rows),
            "Total Length of Fwd Packets": rng.integers(40, 12_000, size=rows),
            "Total Length of Bwd Packets": rng.integers(40, 12_000, size=rows),
            "Flow Packets/s": cic_rate,
            "Fwd Packet Length Mean": rng.uniform(20, 1_500, size=rows),
            "Bwd Packet Length Mean": rng.uniform(20, 1_500, size=rows),
            "Flow IAT Mean": rng.uniform(0, 50_000, size=rows),
            "Flow Bytes/s": rng.lognormal(mean=3.0, sigma=0.8, size=rows),
            "Constant Feature": 7,
            "Label": np.where(attacks, "DDoS", "BENIGN"),
        }
    )
    unsw_rate = rng.lognormal(mean=2.0, sigma=0.7, size=rows)
    unsw_rate[0] = np.inf
    unsw_rate[1] = -np.inf
    unsw_rate[2] = np.nan
    unsw = pd.DataFrame(
        {
            "dur": rng.uniform(0.001, 120.0, size=rows),
            "spkts": rng.integers(1, 100, size=rows),
            "dpkts": rng.integers(1, 90, size=rows),
            "sbytes": rng.integers(40, 12_000, size=rows),
            "dbytes": rng.integers(40, 12_000, size=rows),
            "rate": unsw_rate,
            "smean": rng.uniform(20, 1_500, size=rows),
            "dmean": rng.uniform(20, 1_500, size=rows),
            "sintpkt": rng.uniform(0, 50_000, size=rows),
            "dintpkt": rng.uniform(0, 50_000, size=rows),
            "constant": 0,
            "attack_cat": np.where(attacks, "Exploits", "Normal"),
            "label": attacks.astype("int8"),
        }
    )
    cic_path = cic_dir / "cicids2017_mock.csv"
    unsw_path = unsw_dir / "unsw_nb15_mock.csv"
    cic.to_csv(cic_path, index=False)
    unsw.to_csv(unsw_path, index=False)
    return {"cicids2017": cic_path, "unsw_nb15": unsw_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--rows", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for dataset, path in generate_mock_data(args.output_dir, args.rows, args.seed).items():
        print(f"{dataset}: {path}")


if __name__ == "__main__":
    main()
