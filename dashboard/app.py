"""SOC-style Streamlit dashboard for cross-dataset intrusion detection evaluation.

Launch with:
    streamlit run dashboard/app.py

The dashboard works in two modes:

1.  **Mock / demo mode** (default): synthetic plausible numbers so every
    visualization is populated even without real datasets or trained models.
2.  **Real mode**: reads consolidated result CSVs from ``results/`` when
    the full pipeline has been executed.

Design notes
~~~~~~~~~~~~
*   The dashboard is strictly **read-only**: it never imports training,
    alignment, or preprocessing code, and it never downloads large datasets.
*   Unsupervised alignment (CORAL) does **not** use target labels.  All
    copy and visual labels are careful to state this accurately.
*   Plotly is used for interactive figures; Streamlit columns/metrics give
    a SOC-style overview.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when launched via `streamlit run`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.evaluation import (
    CORE_METRICS,
    MODE_ALIGNED,
    MODE_RAW,
    MODE_SAME,
    ablation_summary,
    class_balance_table,
    compute_grr_table,
    consolidate_results,
    generate_mock_results,
    metric_delta_summary,
)
from src.config import RESULTS_DIR

# ── Page configuration ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cross-Dataset NIDS Evaluation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading evaluation results…")
def load_results() -> tuple[pd.DataFrame, bool]:
    """Try to load real results; fall back to mock data.

    Returns
    -------
    (DataFrame, is_mock) : tuple
        The consolidated result table and a flag indicating whether mock
        data is being used.
    """
    real_path = RESULTS_DIR / "consolidated_results.csv"
    if real_path.is_file():
        try:
            df = pd.read_csv(real_path)
            return df, False
        except Exception as exc:
            st.warning(f"Could not parse {real_path}: {exc}.  Falling back to mock data.")

    return generate_mock_results(), True


data, is_mock = load_results()

# ── Sidebar ───────────────────────────────────────────────────────────────

st.sidebar.title("🛡️ NIDS Evaluator")

if is_mock:
    st.sidebar.warning(
        "**Demo mode — synthetic data.**  "
        "Run the full pipeline via `python main.py run` to generate real results.  "
        "No large datasets are downloaded by the dashboard."
    )
else:
    st.sidebar.success("Loaded real evaluation results.")

# Filters
available_models = sorted(data["model"].unique())
selected_models = st.sidebar.multiselect(
    "Models",
    options=available_models,
    default=available_models,
)

available_modes = sorted(data["evaluation_mode"].unique())
selected_modes = st.sidebar.multiselect(
    "Evaluation modes",
    options=available_modes,
    default=available_modes,
)

metric_choice = st.sidebar.selectbox(
    "Primary metric for charts",
    options=list(CORE_METRICS),
    index=list(CORE_METRICS).index("macro_f1"),
)

filtered = data[
    data["model"].isin(selected_models)
    & data["evaluation_mode"].isin(selected_modes)
].copy()

# ── Header ────────────────────────────────────────────────────────────────

st.title("Cross-Dataset Intrusion Detection — Evaluation Dashboard")
st.markdown(
    "Bidirectional transfer evaluation between **CICIDS-2017** and "
    "**UNSW-NB15** with unsupervised domain alignment (CORAL).  "
    "Target labels are **never** used during alignment — they are only "
    "read to compute evaluation metrics after predictions are generated."
)

if is_mock:
    st.info(
        "ℹ️ You are viewing **synthetic demo data**.  "
        "The numbers are plausible but fabricated for dashboard development.  "
        "Real evaluation results will appear here after running the pipeline.",
        icon="🔬",
    )

# ── KPI row ───────────────────────────────────────────────────────────────

st.header("Key Performance Indicators")

same_df = filtered[filtered["evaluation_mode"] == MODE_SAME]
raw_df = filtered[filtered["evaluation_mode"] == MODE_RAW]
aligned_df = filtered[filtered["evaluation_mode"] == MODE_ALIGNED]

kpi_cols = st.columns(4)

def _safe_mean(series: pd.Series) -> float | None:
    vals = series.dropna()
    return float(vals.mean()) if len(vals) else None

with kpi_cols[0]:
    val = _safe_mean(same_df[metric_choice]) if metric_choice in same_df.columns else None
    st.metric("Same-dataset", f"{val:.3f}" if val is not None else "—")

with kpi_cols[1]:
    val_raw = _safe_mean(raw_df[metric_choice]) if metric_choice in raw_df.columns else None
    st.metric("Raw cross-dataset", f"{val_raw:.3f}" if val_raw is not None else "—")

with kpi_cols[2]:
    val_aligned = _safe_mean(aligned_df[metric_choice]) if metric_choice in aligned_df.columns else None
    delta = None
    if val_aligned is not None and val_raw is not None:
        delta = f"{val_aligned - val_raw:+.3f}"
    st.metric(
        "Aligned cross-dataset",
        f"{val_aligned:.3f}" if val_aligned is not None else "—",
        delta=delta,
    )

with kpi_cols[3]:
    grr_table = compute_grr_table(data)
    grr_col = f"grr_{metric_choice}"
    if grr_col in grr_table.columns:
        avg_grr = _safe_mean(grr_table[grr_col])
        st.metric(
            "Mean GRR",
            f"{avg_grr:.1%}" if avg_grr is not None else "N/A",
            help="Generalization Recovery Rate: fraction of the same-dataset "
                 "→ raw gap recovered by alignment.  N/A when the denominator "
                 "is zero (raw already equals same-dataset).",
        )
    else:
        st.metric("Mean GRR", "N/A")

# ── Raw vs Aligned comparison ─────────────────────────────────────────────

st.header("Raw vs. Aligned Performance")

cross_df = filtered[
    filtered["evaluation_mode"].isin([MODE_RAW, MODE_ALIGNED])
].copy()

if len(cross_df):
    cross_df["direction"] = (
        cross_df["training_domain"] + " → " + cross_df["test_domain"]
    )
    fig_compare = px.bar(
        cross_df,
        x="model",
        y=metric_choice,
        color="evaluation_mode",
        barmode="group",
        facet_col="direction",
        color_discrete_map={
            MODE_RAW: "#EF553B",
            MODE_ALIGNED: "#00CC96",
        },
        labels={
            metric_choice: metric_choice.replace("_", " ").title(),
            "evaluation_mode": "Mode",
            "model": "Model",
        },
        title=f"{metric_choice.replace('_', ' ').title()} — Raw vs. Aligned",
    )
    fig_compare.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig_compare, use_container_width=True)
else:
    st.info("No cross-dataset results to display with the current filters.")

# ── GRR heatmap ───────────────────────────────────────────────────────────

st.header("Generalization Recovery Rate (GRR)")

st.markdown(
    "GRR measures the fraction of the performance gap between "
    "same-dataset and raw cross-dataset that unsupervised alignment recovers.  \n"
    "**GRR = (aligned − raw) / (same-dataset − raw)**  \n"
    "Values of 1.0 = full recovery; > 1.0 = exceeds same-dataset baseline; "
    "N/A = denominator is zero (no gap to recover)."
)

if len(grr_table):
    grr_table["direction"] = (
        grr_table["training_domain"] + " → " + grr_table["test_domain"]
    )
    grr_metrics = [c for c in grr_table.columns if c.startswith("grr_")]
    if grr_metrics:
        # Pivot for heatmap: rows = model+direction, cols = grr metrics.
        heatmap_data = grr_table.set_index(["model", "direction"])[grr_metrics]
        heatmap_data.columns = [c.replace("grr_", "") for c in heatmap_data.columns]

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns.tolist(),
                y=[f"{m} ({d})" for m, d in heatmap_data.index],
                colorscale="RdYlGn",
                zmin=0,
                zmax=1.5,
                text=np.where(
                    np.isnan(heatmap_data.values.astype(float)),
                    "N/A",
                    np.round(heatmap_data.values.astype(float), 3).astype(str),
                ),
                texttemplate="%{text}",
                hovertemplate="Model: %{y}<br>Metric: %{x}<br>GRR: %{z:.3f}<extra></extra>",
            )
        )
        fig_heat.update_layout(
            title="GRR Heatmap — Recovery by Model, Direction, and Metric",
            xaxis_title="Metric",
            yaxis_title="Model (Direction)",
            height=max(350, 60 * len(heatmap_data)),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info(
        "GRR cannot be computed — aligned cross-dataset results are "
        "required alongside same-dataset baselines and raw transfer results."
    )

# ── Ablation summary ─────────────────────────────────────────────────────

st.header("Ablation Summary")

st.markdown(
    "Mean metric values grouped by model and evaluation mode.  "
    "This shows how each pipeline stage — baseline, raw transfer, "
    "unsupervised alignment — contributes to the final performance."
)

try:
    abl = ablation_summary(filtered)
    if len(abl):
        fig_abl = px.bar(
            abl,
            x="model",
            y=metric_choice,
            color="evaluation_mode",
            barmode="group",
            color_discrete_map={
                MODE_SAME: "#636EFA",
                MODE_RAW: "#EF553B",
                MODE_ALIGNED: "#00CC96",
            },
            labels={
                metric_choice: metric_choice.replace("_", " ").title(),
                "evaluation_mode": "Mode",
                "model": "Model",
            },
            title=f"Ablation — Mean {metric_choice.replace('_', ' ').title()} by Mode",
        )
        fig_abl.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig_abl, use_container_width=True)

        st.dataframe(abl, use_container_width=True)
except Exception as exc:
    st.warning(f"Could not compute ablation summary: {exc}")

# ── Metric deltas ─────────────────────────────────────────────────────────

st.header("Alignment Improvement (Δ Raw → Aligned)")

st.markdown(
    "Positive deltas mean alignment *improved* the metric.  "
    "Alignment is **unsupervised** — target labels are not seen during adaptation."
)

try:
    deltas = metric_delta_summary(filtered)
    if len(deltas):
        delta_cols = [c for c in deltas.columns if c.startswith("delta_")]
        deltas["direction"] = (
            deltas["training_domain"] + " → " + deltas["test_domain"]
        )

        # Lollipop chart for the selected metric.
        delta_col = f"delta_{metric_choice}"
        if delta_col in deltas.columns:
            fig_delta = go.Figure()
            for _, row in deltas.iterrows():
                label = f"{row['model']} ({row['direction']})"
                val = row[delta_col]
                color = "#00CC96" if (val is not None and val >= 0) else "#EF553B"
                fig_delta.add_trace(
                    go.Bar(
                        x=[val],
                        y=[label],
                        orientation="h",
                        marker_color=color,
                        showlegend=False,
                        hovertemplate=f"{label}: Δ = %{{x:.4f}}<extra></extra>",
                    )
                )
            fig_delta.update_layout(
                title=f"Δ {metric_choice.replace('_', ' ').title()} (Aligned − Raw)",
                xaxis_title="Delta",
                yaxis_title="",
                height=max(300, 40 * len(deltas)),
            )
            fig_delta.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_delta, use_container_width=True)

        st.dataframe(deltas, use_container_width=True)
except Exception as exc:
    st.warning(f"Could not compute metric deltas: {exc}")

# ── Transfer direction deep-dive ──────────────────────────────────────────

st.header("Transfer Direction Detail")

directions = data[
    data["evaluation_mode"].isin([MODE_RAW, MODE_ALIGNED])
]["training_domain"].unique()

if len(directions):
    for src_domain in sorted(directions):
        dir_data = filtered[filtered["training_domain"] == src_domain]
        if not len(dir_data):
            continue
        tgt_domains = dir_data["test_domain"].unique()
        for tgt_domain in sorted(tgt_domains):
            if tgt_domain == src_domain:
                continue
            pair = dir_data[dir_data["test_domain"] == tgt_domain]
            if not len(pair):
                continue

            with st.expander(f"📡 {src_domain} → {tgt_domain}", expanded=False):
                # Radar chart for all metrics.
                radar_metrics = [
                    m for m in CORE_METRICS
                    if m in pair.columns and pair[m].notna().any()
                ]
                if radar_metrics:
                    fig_radar = go.Figure()
                    for mode in [MODE_RAW, MODE_ALIGNED]:
                        mode_data = pair[pair["evaluation_mode"] == mode]
                        if len(mode_data):
                            means = [float(mode_data[m].mean()) for m in radar_metrics]
                            fig_radar.add_trace(
                                go.Scatterpolar(
                                    r=means + [means[0]],
                                    theta=[m.replace("_", " ").title() for m in radar_metrics]
                                          + [radar_metrics[0].replace("_", " ").title()],
                                    fill="toself",
                                    name=mode.replace("_", " ").title(),
                                    opacity=0.6,
                                )
                            )
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(range=[0, 1])),
                        title=f"Metric Radar — {src_domain} → {tgt_domain}",
                        height=400,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                st.dataframe(
                    pair[["model", "evaluation_mode"] + list(CORE_METRICS)],
                    use_container_width=True,
                )
else:
    st.info("No cross-dataset transfer results available.")

# ── Class balance ─────────────────────────────────────────────────────────

st.header("Dataset & Class Balance Context")

st.markdown(
    "Class distribution in train/test splits.  Balanced sampling is applied "
    "during preprocessing (Module 1), but cross-dataset transfers may test "
    "on differently distributed target data."
)

try:
    balance = class_balance_table(filtered)
    if len(balance):
        fig_bal = px.bar(
            balance.drop_duplicates(
                subset=["dataset", "evaluation_mode"]
            ).melt(
                id_vars=["dataset", "evaluation_mode"],
                value_vars=["train_benign", "train_attack", "test_benign", "test_attack"],
                var_name="split_class",
                value_name="count",
            ),
            x="dataset",
            y="count",
            color="split_class",
            barmode="stack",
            facet_col="evaluation_mode",
            title="Class Counts by Dataset and Mode",
            color_discrete_map={
                "train_benign": "#636EFA",
                "train_attack": "#EF553B",
                "test_benign": "#00CC96",
                "test_attack": "#FFA15A",
            },
        )
        st.plotly_chart(fig_bal, use_container_width=True)

        st.dataframe(balance, use_container_width=True)
except Exception as exc:
    st.warning(f"Could not render class balance: {exc}")

# ── Raw results table ─────────────────────────────────────────────────────

st.header("Full Results Table")

st.dataframe(filtered, use_container_width=True, height=400)

csv_export = filtered.to_csv(index=False)
st.download_button(
    "📥 Download as CSV",
    data=csv_export,
    file_name="nids_evaluation_results.csv",
    mime="text/csv",
)

# ── Footer ────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Module 4 — Evaluation & Dashboard  |  "
    "Cross-Dataset Intrusion Detection B.Tech Project  |  "
    "Md Sabique Huda"
)
