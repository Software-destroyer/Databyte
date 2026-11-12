"""
divergence.py  --  Module 3 (Bamelaai Marbaniang)
Distribution-divergence metrics that explain WHY cross-dataset transfer fails.

Metrics (per Section 5.3 of the project context):
    * Wasserstein distance (per feature)
    * KL divergence        (per feature, 50-bin discretisation)
    * MMD with Gaussian RBF kernel (global, one number)

All functions take two *mapped* DataFrames (output of feature_mapping.apply_mapping)
and compare their common numeric feature columns. The binary "label" column is
always ignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import pairwise_kernels

EPSILON = 1e-10          # avoids log(0) in KL
LABEL_COL = "label"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _common_features(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    """Numeric columns present in both frames (excluding the label), in df_a order."""
    cols = [
        c for c in df_a.columns
        if c != LABEL_COL and c in df_b.columns
        and pd.api.types.is_numeric_dtype(df_a[c])
        and pd.api.types.is_numeric_dtype(df_b[c])
    ]
    if not cols:
        raise ValueError("The two DataFrames share no numeric feature columns.")
    return cols


def _clean(series: pd.Series) -> np.ndarray:
    """float array with NaN / +-Inf removed."""
    arr = series.to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


# --------------------------------------------------------------------------- #
# 1. Wasserstein distance
# --------------------------------------------------------------------------- #
def compute_wasserstein(
    df_a: pd.DataFrame, df_b: pd.DataFrame, normalize: bool = False
) -> pd.DataFrame:
    """
    Per-feature 1-D Wasserstein (Earth Mover's) distance between df_a and df_b.

    Parameters
    ----------
    normalize : if True, each feature is divided by its pooled standard
        deviation first. Raw distances depend on feature units (bytes vs.
        packets), so use normalize=True when you want to *rank* features.

    Returns
    -------
    DataFrame [feature, wasserstein_dist], sorted descending by distance.
    """
    rows = []
    for col in _common_features(df_a, df_b):
        a, b = _clean(df_a[col]), _clean(df_b[col])
        if len(a) == 0 or len(b) == 0:
            rows.append((col, np.nan))
            continue
        if normalize:
            std = np.concatenate([a, b]).std()
            if std > 0:
                a, b = a / std, b / std
        rows.append((col, float(wasserstein_distance(a, b))))
    out = pd.DataFrame(rows, columns=["feature", "wasserstein_dist"])
    return out.sort_values("wasserstein_dist", ascending=False, na_position="last") \
              .reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. KL divergence
# --------------------------------------------------------------------------- #
def compute_kl_divergence(
    df_a: pd.DataFrame, df_b: pd.DataFrame, bins: int = 50
) -> pd.DataFrame:
    """
    Per-feature KL(P_a || P_b) after discretising into `bins` shared bins.

    Bin edges span the pooled 0.5th-99.5th percentile range; values outside it
    are clipped into the end bins. (Network features are heavy-tailed -- with
    plain min/max edges almost everything lands in bin 0 and KL is ~0.)
    EPSILON is added to every bin before normalising to avoid log(0).

    Returns
    -------
    DataFrame [feature, kl_divergence], sorted descending.
    """
    rows = []
    for col in _common_features(df_a, df_b):
        a, b = _clean(df_a[col]), _clean(df_b[col])
        if len(a) == 0 or len(b) == 0:
            rows.append((col, np.nan))
            continue
        pooled = np.concatenate([a, b])
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        if hi <= lo:                       # constant feature in both -> no divergence
            rows.append((col, 0.0))
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(np.clip(a, lo, hi), bins=edges)
        q, _ = np.histogram(np.clip(b, lo, hi), bins=edges)
        p = p / p.sum() + EPSILON
        q = q / q.sum() + EPSILON
        p, q = p / p.sum(), q / q.sum()
        rows.append((col, float(np.sum(p * np.log(p / q)))))
    out = pd.DataFrame(rows, columns=["feature", "kl_divergence"])
    return out.sort_values("kl_divergence", ascending=False, na_position="last") \
              .reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. MMD
# --------------------------------------------------------------------------- #
def compute_mmd(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    sample_size: int = 5000,
    random_state: int = 42,
) -> float:
    """
    Global (multivariate) squared MMD with a Gaussian RBF kernel.

    MMD^2 = mean K(a,a') + mean K(b,b') - 2 mean K(a,b)      (biased estimator)

    Steps: sample up to `sample_size` rows per frame, z-score every feature with
    pooled statistics (so no feature dominates), set the RBF bandwidth with the
    median heuristic, then evaluate via sklearn.pairwise_kernels.
    """
    cols = _common_features(df_a, df_b)
    A = df_a[cols].replace([np.inf, -np.inf], np.nan).dropna()
    B = df_b[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if A.empty or B.empty:
        return float("nan")

    rng = np.random.RandomState(random_state)
    A = A.sample(n=min(sample_size, len(A)), random_state=rng).to_numpy(np.float64)
    B = B.sample(n=min(sample_size, len(B)), random_state=rng).to_numpy(np.float64)

    pooled = np.vstack([A, B])
    mu, sd = pooled.mean(axis=0), pooled.std(axis=0)
    sd[sd == 0] = 1.0
    A, B = (A - mu) / sd, (B - mu) / sd

    # median heuristic on a small subsample of the pooled data
    sub = np.vstack([A, B])
    sub = sub[rng.choice(len(sub), size=min(1000, len(sub)), replace=False)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / (2.0 * med)

    k_aa = pairwise_kernels(A, A, metric="rbf", gamma=gamma).mean()
    k_bb = pairwise_kernels(B, B, metric="rbf", gamma=gamma).mean()
    k_ab = pairwise_kernels(A, B, metric="rbf", gamma=gamma).mean()
    return float(max(k_aa + k_bb - 2.0 * k_ab, 0.0))


# --------------------------------------------------------------------------- #
# 4. Combined report
# --------------------------------------------------------------------------- #
def divergence_report(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str = "A",
    name_b: str = "B",
    normalize: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run Wasserstein + KL + MMD and rank the most divergent features.

    Returns
    -------
    {"wasserstein": DataFrame, "kl": DataFrame, "mmd": float,
     "top5_divergent_features": list[str],
     "names": (name_a, name_b)}

    `normalize=True` makes Wasserstein scale-free so the top-5 ranking is
    meaningful. The top-5 is the top-5 by average rank of Wasserstein and KL.
    """
    w = compute_wasserstein(df_a, df_b, normalize=normalize)
    kl = compute_kl_divergence(df_a, df_b)
    mmd = compute_mmd(df_a, df_b)

    ranks = pd.concat(
        [
            w.set_index("feature")["wasserstein_dist"].rank(ascending=False),
            kl.set_index("feature")["kl_divergence"].rank(ascending=False),
        ],
        axis=1,
    ).mean(axis=1)
    top5 = ranks.sort_values().head(5).index.tolist()

    if verbose:
        print(f"[divergence] {name_a} vs {name_b}: MMD^2={mmd:.4f}, top-5={top5}")

    return {
        "wasserstein": w,
        "kl": kl,
        "mmd": mmd,
        "top5_divergent_features": top5,
        "names": (name_a, name_b),
    }
