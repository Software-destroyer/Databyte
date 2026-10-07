"""Semantic Feature Mapping module for Databyte.

Aligns behavioral features from CICIDS2017 and UNSW-NB15 into a shared
canonical schema with identical column ordering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    PARQUET_COMPRESSION,
    SEMANTIC_FEATURE_MAPPING,
)

LOGGER = logging.getLogger(__name__)


def get_canonical_features(include_label: bool = False) -> list[str]:
    """Return the ordered list of canonical feature names.

    Args:
        include_label: If True, appends 'label' at the end.

    Returns:
        List of canonical column names.
    """
    features = [k for k in SEMANTIC_FEATURE_MAPPING.keys() if k != "label"]
    if include_label:
        features.append("label")
    return features


class FeatureMapper:
    """Map cleaned CICIDS2017 and UNSW-NB15 datasets to canonical features."""

    def __init__(
        self,
        mapping: dict[str, dict[str, str | Sequence[str]]] | None = None,
        output_dir: str | Path = PROCESSED_DATA_DIR,
    ) -> None:
        self.mapping = mapping or SEMANTIC_FEATURE_MAPPING
        self.output_dir = Path(output_dir)
        self.canonical_features = [k for k in self.mapping.keys() if k != "label"]

    @staticmethod
    def _normalize_dataset_name(dataset: str) -> str:
        key = dataset.casefold().replace("-", "_").strip()
        if key in {"cic", "cicids", "cicids2017"}:
            return "cicids2017"
        if key in {"unsw", "unsw_nb15", "unswnb15"}:
            return "unsw_nb15"
        raise ValueError(
            f"Unsupported dataset name '{dataset}'. Expected 'cicids2017' or 'unsw_nb15'."
        )

    def map_dataframe(
        self,
        df: pd.DataFrame,
        dataset: str,
    ) -> pd.DataFrame:
        """Map DataFrame columns to the canonical schema.

        Args:
            df: Cleaned input DataFrame (from Preprocessor or Parquet).
            dataset: Name of dataset ('cicids2017' or 'unsw_nb15').

        Returns:
            pd.DataFrame: Canonical DataFrame with guaranteed column ordering
                          and 'label' as the final column.
        """
        dataset_key = self._normalize_dataset_name(dataset)
        columns_lookup = {col.casefold().strip(): col for col in df.columns}

        mapped_series: dict[str, pd.Series] = {}

        for canonical_name in self.canonical_features:
            spec = self.mapping[canonical_name].get(dataset_key)
            if spec is None:
                raise ValueError(
                    f"No mapping rule defined for '{canonical_name}' in dataset '{dataset_key}'."
                )

            # Handle single field or alternative fields tuple
            candidates = [spec] if isinstance(spec, str) else list(spec)
            matched_col = None
            for cand in candidates:
                cand_clean = cand.casefold().strip()
                if cand_clean in columns_lookup:
                    matched_col = columns_lookup[cand_clean]
                    break

            if matched_col is None:
                raise KeyError(
                    f"Required column for canonical feature '{canonical_name}' "
                    f"(candidates: {candidates}) not found in input columns: {list(df.columns)}"
                )

            values = pd.to_numeric(df[matched_col], errors="coerce").astype(np.float64)
            mapped_series[canonical_name] = values

        # Resolve label column
        label_spec = self.mapping.get("label", {}).get(dataset_key, "label")
        label_candidates = [label_spec] if isinstance(label_spec, str) else list(label_spec)
        label_candidates.extend(["label", "Label"])

        matched_label_col = None
        for cand in label_candidates:
            cand_clean = cand.casefold().strip()
            if cand_clean in columns_lookup:
                matched_label_col = columns_lookup[cand_clean]
                break

        if matched_label_col is None:
            raise KeyError(
                f"Label column not found in input columns: {list(df.columns)}"
            )

        labels = pd.to_numeric(df[matched_label_col], errors="coerce").fillna(0).astype("int8")
        mapped_series["label"] = labels

        # Ensure exact canonical column ordering
        ordered_columns = [*self.canonical_features, "label"]
        aligned_df = pd.DataFrame(mapped_series, index=df.index)[ordered_columns]
        return aligned_df

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        dataset: str | None = None,
    ) -> Path:
        """Load Parquet/CSV, map to canonical features, and export aligned Parquet."""
        source = Path(input_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")

        dataset_key = dataset or ("cicids2017" if "cic" in source.stem.lower() else "unsw_nb15")
        norm_key = self._normalize_dataset_name(dataset_key)

        if source.suffix == ".parquet":
            df = pd.read_parquet(source)
        else:
            df = pd.read_csv(source, low_memory=False)

        aligned_df = self.map_dataframe(df, norm_key)

        if output_path is None:
            filename = (
                "cic_aligned.parquet"
                if norm_key == "cicids2017"
                else "unsw_aligned.parquet"
            )
            destination = self.output_dir / filename
        else:
            destination = Path(output_path)

        destination.parent.mkdir(parents=True, exist_ok=True)
        aligned_df.to_parquet(destination, index=False, compression=PARQUET_COMPRESSION)

        LOGGER.info(
            "Exported %d aligned rows to %s with columns: %s",
            len(aligned_df), destination, list(aligned_df.columns),
        )
        return destination


def map_features(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Convenience function to align a single DataFrame to canonical schema."""
    mapper = FeatureMapper()
    return mapper.map_dataframe(df, dataset)
