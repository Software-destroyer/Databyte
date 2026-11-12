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
RBF_SVM_MAX_TRAIN_ROWS = 10_000

# Shared numeric flow semantics. CIC Flow Duration is converted to seconds
# by FeatureMapper to match UNSW dur; other fields retain common units.
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
        "unsw_nb15": ("smean", "smeansz"),
    },
    "backward_packet_length_mean": {
        "cicids2017": "Bwd Packet Length Mean",
        "unsw_nb15": ("dmean", "dmeansz"),
    },
    "label": {"cicids2017": "Label", "unsw_nb15": "label"},
}

MODEL_HYPERPARAMETERS = {
    "decision_tree": {
        "criterion": "gini",
        "max_depth": 20,
        "random_state": RANDOM_SEED,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 20,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
    "logistic_regression": {
        "penalty": "l2",
        "max_iter": 1_000,
        "random_state": RANDOM_SEED,
    },
    "svm_linear": {"kernel": "linear", "C": 1.0, "random_state": RANDOM_SEED},
    "svm_rbf": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "random_state": RANDOM_SEED},
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
}
