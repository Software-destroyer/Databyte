"""Streaming cleaning and balanced sampling for CICIDS2017 and UNSW-NB15."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from src.config import CSV_CHUNK_SIZE, PARQUET_COMPRESSION, PROCESSED_DATA_DIR, RANDOM_SEED, SUBSAMPLE_SIZE

LOGGER = logging.getLogger(__name__)
_MAX_STAT_SAMPLE = 100_000
UNSW_NB15_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb",
    "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt",
    "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]


class DataPreprocessor:
    """Clean CSV file(s) in bounded chunks and write a stratified Parquet sample.

    Numeric medians and the 99.9th percentile are estimated from a bounded,
    deterministic per-column sample (at most 100,000 finite values). This keeps
    memory use independent of raw dataset size. Output is class-balanced and
    capped at ``sample_size``; when a class is smaller, both classes are capped
    to the minority count without replacement.
    """

    def __init__(self, chunk_size: int = 50_000, sample_size: int = SUBSAMPLE_SIZE,
                 random_state: int = RANDOM_SEED, output_dir: str | Path = PROCESSED_DATA_DIR,
                 infinity_quantile: float = 0.999) -> None:
        if chunk_size < 1 or sample_size < 2:
            raise ValueError("chunk_size must be positive and sample_size at least 2")
        if not 0 < infinity_quantile <= 1:
            raise ValueError("infinity_quantile must be in (0, 1]")
        self.chunk_size = int(chunk_size)
        self.sample_size = int(sample_size)
        self.random_state = int(random_state)
        self.output_dir = Path(output_dir)
        self.infinity_quantile = float(infinity_quantile)

    @staticmethod
    def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
        frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
        return frame

    @staticmethod
    def _resolve_label_column(columns: Iterable[str]) -> str:
        lookup = {str(c).strip().casefold(): str(c) for c in columns}
        if "label" not in lookup:
            raise ValueError("CSV must contain a Label/label column")
        return lookup["label"]

    @staticmethod
    def _encode_labels(series: pd.Series, dataset: str | None) -> pd.Series:
        key = (dataset or "").casefold().replace("-", "_")
        if key in {"unsw", "unsw_nb15", "unswnb15"}:
            values = pd.to_numeric(series, errors="coerce")
            if values.isna().any() or not values.isin([0, 1]).all():
                raise ValueError("UNSW-NB15 label must contain only binary 0/1 values")
            return values.astype("int8")
        if key in {"cic", "cicids", "cicids2017"}:
            if series.isna().any():
                raise ValueError("CICIDS2017 Label contains missing values")
            return (~series.astype(str).str.strip().str.casefold().eq("benign")).astype("int8")
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all() and numeric.isin([0, 1]).all():
            return numeric.astype("int8")
        if series.isna().any():
            raise ValueError("Label contains missing values")
        return (~series.astype(str).str.strip().str.casefold().isin({"benign", "normal"})).astype("int8")

    @staticmethod
    def _features(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
        features = frame.drop(columns=[label_column])
        features = features.loc[:, ~features.columns.astype(str).str.match(r"^Unnamed:\s*\d+$", case=False)]
        # Network datasets include categorical metadata. The shared model uses
        # numeric flow features; discard nonnumeric columns instead of turning
        # them into all-NaN columns that would silently become zero.
        numeric: dict[str, pd.Series] = {}
        for col in features.columns:
            converted = pd.to_numeric(features[col], errors="coerce")
            if converted.notna().any() or pd.api.types.is_numeric_dtype(features[col]):
                numeric[col] = converted
        return pd.DataFrame(numeric, index=features.index)

    @staticmethod
    def _resolve_sources(value: str | Path | Iterable[str | Path]) -> list[Path]:
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(f"Input CSV path does not exist: {path}")
            if path.is_dir():
                paths = sorted(p for p in path.glob("*.csv") if p.is_file())
                # Prefer the four official raw UNSW partitions when present;
                # do not mix in features metadata, ground-truth tables, or a
                # second copy of the official train/test split.
                raw_parts = [p for p in paths if re.search(r"unsw[-_]nb15[_-][1-4]\.csv$", p.name, re.IGNORECASE)]
                if raw_parts:
                    paths = raw_parts
                else:
                    paths = [p for p in paths if not any(token in p.name.casefold() for token in ("features", "_gt", "list_events"))]
            elif path.is_file():
                paths = [path]
            else:
                paths = []
        else:
            paths = [Path(p) for p in value]
            missing = [str(p) for p in paths if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"Input CSV file(s) not found: {missing}")
        if not paths:
            raise FileNotFoundError(f"No CSV files found at {value}")
        return paths

    def _iter_csv(self, sources: list[Path]) -> Iterator[pd.DataFrame]:
        for source in sources:
            # Official UNSW raw partitions commonly have no header row. Detect
            # the standard 49-column layout and attach its published schema.
            columns = list(pd.read_csv(source, nrows=0, low_memory=False).columns)
            has_label = any(str(c).strip().casefold() == "label" for c in columns)
            if not has_label and len(columns) == len(UNSW_NB15_COLUMNS):
                reader = pd.read_csv(source, header=None, names=UNSW_NB15_COLUMNS,
                                     chunksize=self.chunk_size, low_memory=False)
            else:
                reader = pd.read_csv(source, chunksize=self.chunk_size, low_memory=False)
            for chunk in reader:
                yield self._normalise_columns(chunk)

    def _collect_statistics(self, sources: list[Path], dataset: str | None):
        rng = np.random.default_rng(self.random_state)
        class_counts = {0: 0, 1: 0}
        samples: dict[str, np.ndarray] = {}
        mins: dict[str, float] = {}
        maxs: dict[str, float] = {}
        label_column = None
        for chunk in self._iter_csv(sources):
            label_column = self._resolve_label_column(chunk.columns)
            labels = self._encode_labels(chunk[label_column], dataset)
            class_counts[0] += int((labels == 0).sum())
            class_counts[1] += int((labels == 1).sum())
            for col in self._features(chunk, label_column):
                arr = pd.to_numeric(chunk[col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    continue
                mins[col] = min(mins.get(col, float("inf")), float(finite.min()))
                maxs[col] = max(maxs.get(col, float("-inf")), float(finite.max()))
                old = samples.get(col, np.empty(0, dtype=np.float64))
                # Keep a bounded sample; sample each chunk before capping so
                # statistics remain robust for large files and fixed chunks.
                if finite.size > 10_000:
                    finite = rng.choice(finite, 10_000, replace=False)
                merged = np.concatenate((old, finite))
                if merged.size > _MAX_STAT_SAMPLE:
                    merged = rng.choice(merged, _MAX_STAT_SAMPLE, replace=False)
                samples[col] = merged
        if label_column is None or not samples:
            raise ValueError("CSV inputs contain no rows or numeric features")
        if not class_counts[0] or not class_counts[1]:
            raise ValueError("Both benign/normal and attack labels are required")
        medians = {c: float(np.median(v)) for c, v in samples.items() if v.size}
        thresholds = {c: float(np.quantile(v, self.infinity_quantile)) for c, v in samples.items() if v.size}
        kept = [c for c in samples if c in medians and mins[c] != maxs[c]]
        if not kept:
            raise ValueError("All numeric features are constant or entirely missing")
        return kept, medians, thresholds, class_counts

    @staticmethod
    def _balanced_targets(counts: dict[int, int], sample_size: int) -> dict[int, int]:
        per_class = min(counts[0], counts[1], sample_size // 2)
        if per_class == 0:
            raise ValueError("Both classes need at least one row")
        return {0: per_class, 1: per_class}

    def _clean_chunk(self, chunk: pd.DataFrame, dataset: str | None, kept: list[str],
                     medians: dict[str, float], thresholds: dict[str, float]) -> pd.DataFrame:
        label_column = self._resolve_label_column(chunk.columns)
        labels = self._encode_labels(chunk[label_column], dataset)
        features = self._features(chunk, label_column).reindex(columns=kept)
        for col in kept:
            values = pd.to_numeric(features[col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
            values[np.isinf(values)] = thresholds[col]
            values[np.isnan(values)] = medians[col]
            features[col] = values
        features["label"] = labels.to_numpy(dtype=np.int8)
        return features

    def process_file(self, csv_path: str | Path | Iterable[str | Path],
                     output_path: str | Path | None = None, dataset: str | None = None) -> Path:
        sources = self._resolve_sources(csv_path)
        dataset_key = (dataset or sources[0].parent.name or sources[0].stem).casefold().replace("-", "_")
        filename = "cic_cleaned.parquet" if dataset_key in {"cic", "cicids", "cicids2017"} else "unsw_cleaned.parquet"
        destination = Path(output_path) if output_path else self.output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        kept, medians, thresholds, counts = self._collect_statistics(sources, dataset)
        targets = self._balanced_targets(counts, self.sample_size)
        rng = np.random.default_rng(self.random_state)
        seen = {0: 0, 1: 0}
        reservoirs: dict[int, list[dict[str, object]]] = {0: [], 1: []}
        for chunk in self._iter_csv(sources):
            cleaned = self._clean_chunk(chunk, dataset, kept, medians, thresholds)
            for class_id in (0, 1):
                rows = cleaned.loc[cleaned.label == class_id].reset_index(drop=True)
                old_seen = seen[class_id]
                reservoir = reservoirs[class_id]
                target = targets[class_id]
                start = 0
                if len(reservoir) < target:
                    initial = min(target - len(reservoir), len(rows))
                    reservoir.extend(rows.iloc[:initial].to_dict(orient="records"))
                    start = initial
                if start < len(rows):
                    global_indices = old_seen + np.arange(start, len(rows), dtype=np.int64)
                    replacements = rng.integers(0, global_indices + 1)
                    accepted = np.flatnonzero(replacements < target)
                    for local_index in accepted:
                        reservoir[int(replacements[local_index])] = rows.iloc[[start + int(local_index)]].to_dict(orient="records")[0]
                seen[class_id] += len(rows)
        if any(len(reservoirs[c]) != targets[c] for c in (0, 1)):
            raise RuntimeError("Could not collect the requested stratified sample")
        result = pd.concat([pd.DataFrame(reservoirs[c], columns=[*kept, "label"]) for c in (0, 1)], ignore_index=True)
        result = result.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
        try:
            result.to_parquet(temporary, index=False, compression=PARQUET_COMPRESSION)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        LOGGER.info("Wrote %s rows (%s per class) from %s CSV file(s) to %s", len(result), targets[0], len(sources), destination)
        return destination

    def process_cicids2017(self, csv_path, output_path=None) -> Path:
        return self.process_file(csv_path, output_path, dataset="cicids2017")

    def process_unsw_nb15(self, csv_path, output_path=None) -> Path:
        return self.process_file(csv_path, output_path, dataset="unsw_nb15")

    def process_datasets(self, cicids_csv, unsw_csv) -> dict[str, Path]:
        return {"cicids2017": self.process_cicids2017(cicids_csv), "unsw_nb15": self.process_unsw_nb15(unsw_csv)}
