"""Streaming cleaning and balanced sampling for network-flow CSV datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.config import (
    CSV_CHUNK_SIZE,
    PARQUET_COMPRESSION,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SUBSAMPLE_SIZE,
)

LOGGER = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess CICIDS2017/UNSW-NB15 CSV files using bounded chunk reads.

    The output is class-balanced (equal benign and attack examples), capped at
    ``sample_size``. Feature statistics are accumulated by chunk and only the
    selected sample is kept in memory for the final Parquet write.
    """

    def __init__(
        self,
        chunk_size: int = 50_000,
        sample_size: int = SUBSAMPLE_SIZE,
        random_state: int = RANDOM_SEED,
        output_dir: str | Path = PROCESSED_DATA_DIR,
        infinity_quantile: float = 0.999,
    ) -> None:
        if chunk_size <= 0 or sample_size <= 0:
            raise ValueError("chunk_size and sample_size must be positive")
        if not 0 < infinity_quantile <= 1:
            raise ValueError("infinity_quantile must be in (0, 1]")
        self.chunk_size = int(chunk_size)
        self.sample_size = int(sample_size)
        self.random_state = int(random_state)
        self.output_dir = Path(output_dir)
        self.infinity_quantile = float(infinity_quantile)

    @staticmethod
    def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
        frame.columns = [str(name).strip().lstrip("\ufeff") for name in frame.columns]
        return frame

    @staticmethod
    def _resolve_label_column(columns: list[str]) -> str:
        lookup = {name.casefold(): name for name in columns}
        for candidate in ("label",):
            if candidate in lookup:
                return lookup[candidate]
        raise ValueError("CSV must contain a 'Label' or 'label' column")

    @staticmethod
    def _encode_labels(series: pd.Series, dataset: str | None) -> pd.Series:
        if dataset and dataset.casefold() in {"unsw", "unsw_nb15", "unsw-nb15"}:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.isna().any() or not numeric.isin([0, 1]).all():
                raise ValueError("UNSW-NB15 'label' column must contain only 0 and 1")
            return numeric.astype("int8")
        if dataset and dataset.casefold() in {"cic", "cicids", "cicids2017"}:
            if series.isna().any():
                raise ValueError("CICIDS2017 Label column contains missing values")
            return (~series.astype(str).str.strip().str.casefold().eq("benign")).astype("int8")

        # Infer CICIDS text labels versus UNSW binary labels when dataset is
        # omitted. Numeric 0/1 labels have the same mapping in both datasets.
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all() and numeric.isin([0, 1]).all():
            return numeric.astype("int8")
        if series.isna().any():
            raise ValueError("Label column contains missing values")
        return (~series.astype(str).str.strip().str.casefold().eq("benign")).astype("int8")

    @staticmethod
    def _features(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
        features = frame.drop(columns=[label_column]).copy()
        features = features.loc[
            :, ~features.columns.astype(str).str.match(r"^Unnamed:\s*\d+$", case=False)
        ]
        for column in features.columns:
            if not pd.api.types.is_numeric_dtype(features[column]):
                features[column] = pd.to_numeric(features[column], errors="coerce")
        return features

    def _iter_csv(self, csv_path: Path) -> Iterator[pd.DataFrame]:
        for chunk in pd.read_csv(csv_path, chunksize=self.chunk_size, low_memory=False):
            yield self._normalise_columns(chunk)

    def _collect_statistics(self, csv_path: Path, dataset: str | None):
        counts: dict[str, int] = {}
        sums: dict[str, float] = {}
        sums_sq: dict[str, float] = {}
        finite_values: dict[str, list[np.ndarray]] = {}
        class_counts = {0: 0, 1: 0}
        label_column: str | None = None

        for chunk in self._iter_csv(csv_path):
            label_column = self._resolve_label_column(list(chunk.columns))
            labels = self._encode_labels(chunk[label_column], dataset)
            class_counts[0] += int(labels.eq(0).sum())
            class_counts[1] += int(labels.eq(1).sum())
            for column in self._features(chunk, label_column).columns:
                raw = pd.to_numeric(chunk[column], errors="coerce").to_numpy(
                    dtype=np.float64, na_value=np.nan
                )
                valid = raw[np.isfinite(raw)]
                if column not in counts:
                    counts[column] = 0
                    sums[column] = 0.0
                    sums_sq[column] = 0.0
                    finite_values[column] = []
                if valid.size:
                    counts[column] += valid.size
                    sums[column] += float(valid.sum(dtype=np.float64))
                    sums_sq[column] += float(np.square(valid).sum(dtype=np.float64))
                    finite_values[column].append(valid)

        if label_column is None or not counts:
            raise ValueError("CSV contains no data rows or numeric feature columns")
        if not class_counts[0] or not class_counts[1]:
            raise ValueError("Both benign/normal and attack classes are required")

        medians: dict[str, float] = {}
        thresholds: dict[str, float] = {}
        nonzero_variance: list[str] = []
        for column in counts:
            values = np.concatenate(finite_values[column]) if finite_values[column] else np.array([])
            medians[column] = float(np.median(values)) if values.size else 0.0
            count = counts[column]
            mean = sums[column] / count if count else 0.0
            variance = max(sums_sq[column] / count - mean * mean, 0.0) if count else 0.0
            if variance > 0.0:
                nonzero_variance.append(column)
            if values.size:
                thresholds[column] = float(np.quantile(values, self.infinity_quantile))
        if not nonzero_variance:
            raise ValueError("All feature columns are constant")
        return label_column, nonzero_variance, medians, thresholds, class_counts

    @staticmethod
    def _balanced_targets(class_counts: dict[int, int], sample_size: int) -> dict[int, int]:
        per_class = min(class_counts[0], class_counts[1], sample_size // 2)
        if per_class < 1:
            raise ValueError("sample_size must be at least 2 and both classes must be present")
        return {0: per_class, 1: per_class}

    def _clean_chunk(
        self,
        chunk: pd.DataFrame,
        dataset: str | None,
        kept_columns: list[str],
        medians: dict[str, float],
        thresholds: dict[str, float],
    ) -> pd.DataFrame:
        label_column = self._resolve_label_column(list(chunk.columns))
        labels = self._encode_labels(chunk[label_column], dataset)
        features = self._features(chunk, label_column).reindex(columns=kept_columns)
        for column in kept_columns:
            values = pd.to_numeric(features[column], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan, copy=True
            ).copy()
            infinities = np.isinf(values)
            values[infinities] = thresholds.get(column, medians[column])
            missing = np.isnan(values)
            values[missing] = medians[column]
            features[column] = values
        features["label"] = labels.to_numpy(dtype=np.int8)
        return features

    def process_file(
        self,
        csv_path: str | Path,
        output_path: str | Path | None = None,
        dataset: str | None = None,
    ) -> Path:
        """Clean one source CSV, write Parquet, and return its path."""
        source = Path(csv_path)
        if not source.is_file():
            raise FileNotFoundError(f"Input CSV not found: {source}")
        dataset_key = dataset or source.parent.name or source.stem
        normalized_key = dataset_key.casefold().replace("-", "_")
        if output_path is None:
            filename = "cic_cleaned.parquet" if normalized_key in {"cic", "cicids", "cicids2017"} else "unsw_cleaned.parquet"
            destination = self.output_dir / filename
        else:
            destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        label_column, kept_columns, medians, thresholds, class_counts = self._collect_statistics(
            source, dataset
        )
        targets = self._balanced_targets(class_counts, self.sample_size)
        seen = {0: 0, 1: 0}
        rng = np.random.default_rng(self.random_state)

        # Reservoir sampling over cleaned rows preserves an unbiased sample
        # within each class without retaining the full dataset in memory.
        reservoirs: dict[int, list[dict[str, object]]] = {0: [], 1: []}
        for chunk in self._iter_csv(source):
            cleaned = self._clean_chunk(chunk, dataset, kept_columns, medians, thresholds)
            for class_id in (0, 1):
                class_rows = cleaned.loc[cleaned["label"] == class_id].reset_index(drop=True)
                count = len(class_rows)
                if not count:
                    continue
                old_seen = seen[class_id]
                keep_frames = reservoirs[class_id]
                if old_seen < targets[class_id]:
                    initial_count = min(targets[class_id] - old_seen, count)
                    if initial_count:
                        keep_frames.extend(class_rows.iloc[:initial_count].to_dict(orient="records"))
                    remaining_start = initial_count
                else:
                    remaining_start = 0
                # Vectorized reservoir candidate generation for rows after
                # the reservoir fills. Candidate batches are at most one CSV
                # chunk, and replacement decisions remain uniformly random.
                for row_index in range(remaining_start, count):
                    global_index = old_seen + row_index + 1
                    if len(keep_frames) < targets[class_id]:
                        keep_frames.append(class_rows.iloc[row_index].to_dict())
                    else:
                        replacement = int(rng.integers(0, global_index))
                        if replacement < targets[class_id]:
                            keep_frames[replacement] = class_rows.iloc[row_index].to_dict()
                seen[class_id] += count
                reservoirs[class_id] = keep_frames

        pieces = [
            pd.DataFrame(reservoirs[class_id], columns=[*kept_columns, "label"])
            for class_id in (0, 1)
        ]
        if any(len(reservoirs[class_id]) != targets[class_id] for class_id in (0, 1)):
            raise RuntimeError("Could not collect the requested balanced sample")
        result = pd.concat(pieces, ignore_index=True)
        result = result.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
        try:
            result.to_parquet(temporary, index=False, compression=PARQUET_COMPRESSION)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        LOGGER.info(
            "Wrote %d balanced rows (%d per class) from %s to %s",
            len(result), targets[0], source, destination,
        )
        return destination

    def process_cicids2017(self, csv_path: str | Path, output_path: str | Path | None = None) -> Path:
        return self.process_file(csv_path, output_path, dataset="cicids2017")

    def process_unsw_nb15(self, csv_path: str | Path, output_path: str | Path | None = None) -> Path:
        return self.process_file(csv_path, output_path, dataset="unsw_nb15")

    def process_datasets(
        self,
        cicids_csv: str | Path,
        unsw_csv: str | Path,
    ) -> dict[str, Path]:
        """Process both dataset inputs to the requested canonical filenames."""
        return {
            "cicids2017": self.process_cicids2017(cicids_csv),
            "unsw_nb15": self.process_unsw_nb15(unsw_csv),
        }
