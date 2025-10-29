import json
import os
import re
from glob import glob

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# Config
# =========================
PREFERRED_DIRS = ["data", "/mnt/data"]
st.set_page_config(page_title="VLLM Reasoning Comparison", layout="wide")

# =========================
# Helpers
# =========================
def infer_problem_set_from_name(fname: str, parsed: dict | None) -> str:
    name = os.path.basename(fname).lower()
    if re.search(r"[_\-]19([_\-]|\.json$)", name):
        return "19"
    if "79problems" in name or re.search(r"[_\-]79([_\-]|\.json$)", name):
        return "79"
    if parsed and isinstance(parsed, dict) and isinstance(parsed.get("problems"), list):
        n = len(parsed["problems"])
        if n <= 30:
            return "19"
        if n >= 60:
            return "79"
    return "unknown"

def coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

# =========================
# Parsing
# =========================
def parse_file(path: str):
    """
    Returns:
      per_problem_rows: list of dicts with per-problem info (category, difficulty, steps_count)
      per_model_row: dict with per-model aggregates from 'evaluations'/'analysis'
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    model_name = coalesce(obj.get("model"), os.path.splitext(os.path.basename(path))[0])
    problem_set = infer_problem_set_from_name(path, obj)

    # --- Per-problem rows (for steps distribution, raw mixes)
    per_problem_rows = []
    problems = obj.get("problems", [])
    if not isinstance(problems, list):
        problems = []

    for p in problems:
        steps_raw = p.get("solution_steps") or p.get("steps") or p.get("reasoning_steps")
        if isinstance(steps_raw, list):
            steps_count = len(steps_raw)
        elif isinstance(steps_raw, str):
            chunks = [s for s in re.split(r"[\n;•\-]+", steps_raw) if s.strip()]
            steps_count = len(chunks) if chunks else None
        else:
            steps_count = None

        per_problem_rows.append(
            {
                "model": model_name,
                "problem_set": problem_set,
                "category": p.get("category"),
                "difficulty": p.get("difficulty"),
                "steps_count": steps_count,
            }
        )

    # --- Per-model aggregates (comparisons)
    eval0 = None
    if isinstance(obj.get("evaluations"), list) and obj["evaluations"]:
        eval0 = obj["evaluations"][0]

    analysis = obj.get("analysis", {}) if isinstance(obj.get("analysis"), dict | None) else {}

    avg_score = None
    avg_step_score = None
    if isinstance(eval0, dict):
        avg_score = eval0.get("average_score")  # 0..1
        avg_step_score = eval0.get("average_step_score")  # 0..1

    per_model_row = {
        "model": model_name,
        "problem_set": problem_set,
        # prefer analysis.overall_average if present
        "overall_score": coalesce(analysis.get("overall_average"), avg_score),
        "step_accuracy": analysis.get("step_accuracy"),
        "consistency": analysis.get("consistency"),
        "average_step_score": avg_step_score,
        "difficulty_averages": analysis.get("difficulty_averages"),  # dict[str->0..1]
        "category_averages": analysis.get("category_averages"),      # dict[str->0..1]
    }

    return per_problem_rows, per_model_row

@st.cache_data
def load_all():
    files = []
    for d in PREFERRED_DIRS:
        if os.path.isdir(d):
            files.extend(glob(os.path.join(d, "*.json")))
    files = sorted(set(files))
    per_problem_rows, per_model_rows = [], []
    info_rows = []

    for p in files:
        try:
            pp, pm = parse_file(p)
            per_problem_rows.extend(pp)
            per_model_rows.append(pm)
            info_rows.append(
                {
                    "file": os.path.basename(p),
                    "model": pm["model"],
                    "problem_set": pm["problem_set"],
                    "parsed_problems": len(pp),
                    "has_overall_score": pm["overall_score"] is not None,
                }
            )
        except Exception as e:
            info_rows.append({"file": os.path.basename(p), "model": "—", "problem_set": "—", "parsed_problems": 0, "has_overall_score": False})
            st.error(f"❌ Parse error in {p}: {e}")

    df_problems = pd.DataFrame(per_problem_rows)
    df_models = pd.DataFrame(per_model_rows).drop_duplicates(subset=["model", "problem_set"], keep="last")
    df_info = pd.DataFrame(info_rows)
    return df_problems, df_models, df_info

# =========================
# UI
# =========================
st.title("VLLM Reasoning Comparison (19 vs 79)")

with st.expander("ℹ️ What’s plotted?"):
    st.markdown(
        """
- **Overall score** (accuracy proxy): uses `analysis.overall_average` if present, otherwise `evaluations[0].average_score`.  
- **Step accuracy** and **Consistency** come from `analysis`.  
- **Difficulty radar** and **Category heatmap** use the per-group averages in `analysis`.  
- **Steps distribution** is derived from `solution_steps` / `steps`.  
If a metric is missing for some file, that model simply won’t show for that chart.
"""
    )

df_problems, df_models, df_info = load_all()

with st.expander("Loaded files"):
    st.dataframe(df_info, use_container_width=True)

if df_models.empty and df_problems.empty:
    st.warning("No data parsed. Make sure your JSONs are in ./data or /mnt/data.")
    st.stop()

# -------------------------
# Controls
# -------------------------
left, right = st.columns([1, 2], gap="large")

with left:
    ps_choice = st.radio("Problem set", ["both", "19", "79"], index=0, horizontal=True)
    if ps_choice != "both":
        models_ps = df_models[df_models["problem_set"] == ps_choice]
        probs_ps = df_problems[df_problems["problem_set"] == ps_choice]
    else:
        models_ps = df_models.copy()
        probs_ps = df_problems.copy()

    available_models = sorted(models_ps["model"].dropna().unique().tolist())
    default_sel = available_models[: min(len(available_models), 8)]
    picked_models = st.multiselect("Models", available_models, default=default_sel)

with right:
    viz = st.selectbox(
        "Visualization",
        [
            "Overall score (accuracy proxy)",
            "Step accuracy",
            "Consistency",
            "Difficulty radar (averages)",
            "Category heatmap (averages)",
            "Reasoning steps distribution",
        ],
    )

models_ps = models_ps[models_ps["model"].isin(picked_models)]
probs_ps = probs_ps[probs_ps["model"].isin(picked_models)]

# -------------------------
# Visualizations
# -------------------------
def pct(x):
    try:
        return float(x) * 100.0
    except Exception:
        return None

if viz == "Overall score (accuracy proxy)":
    d = models_ps.dropna(subset=["overall_score"]).copy()
    if d.empty:
        st.info("No overall scores found in selected files.")
    else:
        d["overall_score_pct"] = d["overall_score"].apply(pct)
        fig = px.bar(
            d,
            x="model",
            y="overall_score_pct",
            color="problem_set",
            barmode="group",
            title="Overall score by model",
            labels={"overall_score_pct": "Overall score (%)", "model": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

elif viz == "Step accuracy":
    d = models_ps.dropna(subset=["step_accuracy"]).copy()
    if d.empty:
        st.info("No step accuracy found in selected files.")
    else:
        d["step_accuracy_pct"] = d["step_accuracy"].apply(pct)
        fig = px.bar(
            d,
            x="model",
            y="step_accuracy_pct",
            color="problem_set",
            barmode="group",
            title="Step accuracy by model",
            labels={"step_accuracy_pct": "Step accuracy (%)", "model": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

elif viz == "Consistency":
    d = models_ps.dropna(subset=["consistency"]).copy()
    if d.empty:
        st.info("No consistency values found in selected files.")
    else:
        # Consistency is already 0..1; show %
        d["consistency_pct"] = d["consistency"].apply(pct)
        fig = px.bar(
            d,
            x="model",
            y="consistency_pct",
            color="problem_set",
            barmode="group",
            title="Consistency by model",
            labels={"consistency_pct": "Consistency (%)", "model": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

elif viz == "Difficulty radar (averages)":
    d = models_ps.dropna(subset=["difficulty_averages"]).copy()
    if d.empty:
        st.info("No difficulty averages found in selected files.")
    else:
        # explode dict
        rows = []
        for _, r in d.iterrows():
            for diff, val in (r["difficulty_averages"] or {}).items():
                rows.append(
                    {
                        "model": r["model"],
                        "problem_set": r["problem_set"],
                        "difficulty": diff,
                        "value_pct": pct(val),
                    }
                )
        dd = pd.DataFrame(rows)
        if dd.empty:
            st.info("Difficulty averages dicts were empty.")
        else:
            # Use consistent difficulty order
            diff_order = ["Easy", "Medium", "Hard"]
            others = sorted(set(dd["difficulty"]) - set(diff_order))
            difficulties = diff_order + others

            tabs = st.tabs(sorted(dd["problem_set"].dropna().unique()))
            for i, ps in enumerate(sorted(dd["problem_set"].dropna().unique())):
                with tabs[i]:
                    sub = dd[dd["problem_set"] == ps]
                    fig = go.Figure()
                    for m in sorted(sub["model"].unique()):
                        row = (
                            sub[sub["model"] == m]
                            .set_index("difficulty")
                            .reindex(difficulties)["value_pct"]
                            .fillna(0)
                            .tolist()
                        )
                        fig.add_trace(
                            go.Scatterpolar(r=row, theta=difficulties, fill="toself", name=m)
                        )
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=True,
                        title=f"Difficulty averages (%) — Problem set {ps}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif viz == "Category heatmap (averages)":
    d = models_ps.dropna(subset=["category_averages"]).copy()
    if d.empty:
        st.info("No category averages found in selected files.")
    else:
        # explode dict
        rows = []
        for _, r in d.iterrows():
            for cat, val in (r["category_averages"] or {}).items():
                rows.append(
                    {
                        "model": r["model"],
                        "problem_set": r["problem_set"],
                        "category": cat,
                        "value_pct": pct(val),
                    }
                )
        dd = pd.DataFrame(rows)
        if dd.empty:
            st.info("Category averages dicts were empty.")
        else:
            # For each problem_set, separate heatmaps to keep readable
            tabs = st.tabs(sorted(dd["problem_set"].dropna().unique()))
            for i, ps in enumerate(sorted(dd["problem_set"].dropna().unique())):
                with tabs[i]:
                    sub = dd[dd["problem_set"] == ps]
                    pivot = sub.pivot_table(
                        index="category", columns="model", values="value_pct", aggfunc="mean"
                    ).fillna(0)
                    fig = px.imshow(
                        pivot,
                        labels=dict(x="Model", y="Category", color="Avg (%)"),
                        title=f"Category averages (%) — Problem set {ps}",
                        aspect="auto",
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif viz == "Reasoning steps distribution":
    t = probs_ps.dropna(subset=["steps_count"])
    if t.empty:
        st.info("No step counts found (expected `solution_steps` / `steps`).")
    else:
        fig = px.violin(
            t,
            x="model",
            y="steps_count",
            color="problem_set",
            box=True,
            points="all",
            title="Distribution of reasoning steps per model",
            labels={"steps_count": "Step count", "model": ""},
        )
        st.plotly_chart(fig, use_container_width=True)
