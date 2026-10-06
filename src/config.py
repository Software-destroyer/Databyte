"""Central configuration for the cross-dataset intrusion detection project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_SEED = 42
CSV_CHUNK_SIZE = 100_000
DEFAULT_SUBSAMPLE_SIZE = 200_000
SUBSAMPLE_SIZE = DEFAULT_SUBSAMPLE_SIZE
PARQUET_COMPRESSION = "snappy"

# Canonical names map to dataset-specific names. Entries with more than one
# UNSW field are alternatives and are intentionally not combined implicitly.
SEMANTIC_FEATURE_MAPPING = {
    "flow_duration": {"cicids2017": "Flow Duration", "unsw_nb15": "dur"},
    "total_forward_packets": {
        "cicids2017": "Total Fwd Packets",
        "unsw_nb15": "spkts",
    },
    "total_backward_packets": {
        "cicids2017": "Total Backward Packets",
        "unsw_nb15": "dpkts",
    },
    "total_forward_bytes": {
        "cicids2017": "Total Length of Fwd Packets",
        "unsw_nb15": "sbytes",
    },
    "total_backward_bytes": {
        "cicids2017": "Total Length of Bwd Packets",
        "unsw_nb15": "dbytes",
    },
    "flow_packet_rate": {"cicids2017": "Flow Packets/s", "unsw_nb15": "rate"},
    "forward_packet_length_mean": {
        "cicids2017": "Fwd Packet Length Mean",
        "unsw_nb15": "smean",
    },
    "backward_packet_length_mean": {
        "cicids2017": "Bwd Packet Length Mean",
        "unsw_nb15": "dmean",
    },
    "flow_iat_mean": {
        "cicids2017": "Flow IAT Mean",
        "unsw_nb15": ("sintpkt", "dintpkt"),
    },
    "label": {"cicids2017": "Label", "unsw_nb15": "label"},
}

MODEL_HYPERPARAMETERS = {
    "decision_tree": {"criterion": "gini", "random_state": RANDOM_SEED},
    "random_forest": {
        "n_estimators": 100,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
    "logistic_regression": {
        "penalty": "l2",
        "max_iter": 1_000,
        "random_state": RANDOM_SEED,
    },
    "svm_linear": {"kernel": "linear", "probability": True, "random_state": RANDOM_SEED},
    "svm_rbf": {"kernel": "rbf", "probability": True, "random_state": RANDOM_SEED},
    "xgboost": {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "eval_metric": "logloss",
    },
}
