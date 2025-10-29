import json
import os
import re
from glob import glob
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------- Config ----------
DATA_DIR = "data"   # put your JSONs here. You can point this to "/mnt/data" if preferred.
st.set_page_config(page_title="VLLM Reasoning Comparison", layout="wide")

# ---------- Utils ----------
@st.cache_data
def find_json_files(data_dir):
    # Grab both 19- and 79-problem files (by filename hints)
    paths = glob(os.path.join(data_dir, "*.json"))
    return sorted(paths)

def infer_problem_set(filename, parsed):
    """
    Try to infer whether this file is the 19- or 79-problem set.
    Priority:
      1) filename contains '_19_' or '_79'
      2) len(parsed['problems']) if present
    """
    name = os.path.basename(filename).lower()
    if re.search(r"[_\-]19[_\-]|_19\.json$", name):
        return "19"
    if re.search(r"[_\-]79[_\-]|_79\.json$", name) or "79problems" in name:
        return "79"
    # fallback by length if "problems" present
    if isinstance(parsed, dict) and "problems" in parsed and isinstance(parsed["problems"], list):
        n = len(parsed["problems"])
        if n <= 30:
            return "19"
        if n >= 60:
            return "79"
    return "unknown"

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def extract_records(json_obj):
    """
    Normalize various schemas into a list of per-problem records with:
      model, problem_set, category, difficulty, steps_count, is_correct (optional)
    Also returns a model_name string.
    """
    records = []
    # model name
    model_name = safe_get(json_obj, "model", default=None)
    if model_name is None:
        # sometimes embedded in filename only; caller can fill later
        model_name = "Unknown Model"

    # common schema: {"problems": [{...}]}
    problems = safe_get(json_obj, "problems", default=None)

    # alternative schema: {"results":[...]} or similar
    if problems is None:
        if isinstance(json_obj, dict):
            for k in ("results", "items", "data"):
                if k in json_obj and isinstance(json_obj[k], list):
                    problems = json_obj[k]
                    break

    if not problems or not isinstance(problems, list):
        return model_name, records

    for p in problems:
        category = p.get("category")
        difficulty = p.get("difficulty")
        # steps: look for explicit list, or count from a reasoning field
        steps = p.get("solution_steps") or p.get("steps") or p.get("reasoning_steps")
        if isinstance(steps, list):
            steps_count = len(steps)
        elif isinstance(steps, str):
            steps_count = len([s for s in re.split(r"[\n;•-]+", steps) if s.strip()])
        else:
            steps_count = None

        # correctness / pass flag in some evaluation dumps
        # try common keys:
        is_correct = p.get("correct")
        if is_correct is None:
            is_correct = p.get("is_correct")
        if is_correct is None:
            # sometimes "score": 1/0
            score = p.get("score")
            if score is not None:
                try:
                    is_correct = bool(int(score))
                except Exception:
                    is_correct = None

        records.append({
            "category": category,
            "difficulty": difficulty,
            "steps_count": steps_count,
            "is_correct": is_correct
        })
    return model_name, records

@st.cache_data
def load_all(data_dir):
    rows = []
    file_info = []
    for path in find_json_files(data_dir):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            file_info.append((os.path.basename(path), f"❌ Parse error: {e}"))
            continue

        problem_set = infer_problem_set(path, obj)
        model_name, records = extract_records(obj)
        # If model missing, try deriving from filename
        if (not model_name) or model_name == "Unknown Model":
            model_name = os.path.splitext(os.path.basename(path))[0]

        for r in records:
            rows.append({
                "model": model_name,
                "problem_set": problem_set,
                **r
            })

        file_info.append((os.path.basename(path), f"✅ {len(records)} problems, PS={problem_set}, model={model_name}"))
    df = pd.DataFrame(rows)
    info_df = pd.DataFrame(file_info, columns=["file", "status"])
    return df, info_df

# ---------- Load ----------
st.title("VLLM Reasoning Comparison (19 vs 79 problems)")
with st.expander("ℹ️ How this works"):
    st.markdown("""
- Drop your JSON result files into the **data/** folder.  
- The app tries to normalize different schemas (e.g., `problems`, `results`) into a standard table.  
- If a file contains correctness flags (`correct`, `is_correct`, or `score` 1/0), **accuracy** will be computed.
- Otherwise, you still get **category mix**, **difficulty mix**, and **reasoning length** (via step counts).
""")

df, info = load_all(DATA_DIR)
if df.empty:
    st.warning("No parsed problems found. Check your `data/` folder path or file formats.")
    st.dataframe(info, use_container_width=True)
    st.stop()

with st.expander("Loaded files"):
    st.dataframe(info, use_container_width=True)

# ---------- Controls ----------
left, right = st.columns([1, 2])
with left:
    ps_choice = st.radio("Problem set", ["19", "79", "both"], index=2, horizontal=True)
    if ps_choice != "both":
        df_show = df[df["problem_set"] == ps_choice]
    else:
        df_show = df.copy()
    models = sorted(df_show["model"].dropna().unique().tolist())
    picked_models = st.multiselect("Models", models, default=models[:min(len(models), 6)])

with right:
    viz = st.selectbox(
        "Visualization",
        [
            "Accuracy by model",
            "Difficulty mix (stacked)",
            "Category heatmap",
            "Reasoning steps (distribution)",
            "Radar: difficulty share"
        ]
    )

df_show = df_show[df_show["model"].isin(picked_models)]

# ---------- Visuals ----------
def compute_accuracy(frame):
    has_correct = frame["is_correct"].notna().any()
    if not has_correct:
        return None
    out = (frame
           .dropna(subset=["is_correct"])
           .groupby(["model", "problem_set"])["is_correct"]
           .mean()
           .reset_index(name="accuracy"))
    out["accuracy"] *= 100.0
    return out

if viz == "Accuracy by model":
    acc = compute_accuracy(df_show)
    if acc is None or acc.empty:
        st.info("No correctness flags detected in the selected files — cannot compute accuracy.")
    else:
        bars = px.bar(
            acc,
            x="model", y="accuracy",
            color="problem_set",
            barmode="group",
            title="Accuracy by model (mean of is_correct)"
        )
        bars.update_layout(yaxis_title="Accuracy (%)", xaxis_title="")
        st.plotly_chart(bars, use_container_width=True)

elif viz == "Difficulty mix (stacked)":
    temp = (df_show
            .assign(difficulty=df_show["difficulty"].fillna("unknown"))
            .groupby(["model", "problem_set", "difficulty"])
            .size()
            .reset_index(name="count"))
    if temp.empty:
        st.info("No difficulty labels found.")
    else:
        fig = px.bar(
            temp,
            x="model", y="count",
            color="difficulty",
            facet_row="problem_set",
            title="Difficulty mix per model & problem set",
            category_orders={"difficulty": ["Easy", "Medium", "Hard", "unknown"]}
        )
        st.plotly_chart(fig, use_container_width=True)

elif viz == "Category heatmap":
    temp = (df_show
            .assign(category=df_show["category"].fillna("unknown"))
            .groupby(["model", "category"])
            .size()
            .reset_index(name="count"))
    if temp.empty:
        st.info("No category labels found.")
    else:
        pivot = temp.pivot(index="category", columns="model", values="count").fillna(0)
        heat = px.imshow(
            pivot,
            labels=dict(x="Model", y="Category", color="# problems"),
            title="Category coverage by model"
        )
        st.plotly_chart(heat, use_container_width=True)

elif viz == "Reasoning steps (distribution)":
    temp = df_show.dropna(subset=["steps_count"])
    if temp.empty:
        st.info("No step counts found (looking for `solution_steps` or similar).")
    else:
        fig = px.violin(
            temp,
            x="model",
            y="steps_count",
            color="problem_set",
            box=True,
            points="all",
            title="Distribution of reasoning steps per model"
        )
        fig.update_layout(yaxis_title="Step count", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

elif viz == "Radar: difficulty share":
    temp = (df_show
            .assign(difficulty=df_show["difficulty"].fillna("unknown"))
            .groupby(["model", "problem_set", "difficulty"])
            .size()
            .groupby(level=[0,1])
            .apply(lambda s: (s / s.sum()) * 100)
            .reset_index(name="pct"))
    if temp.empty:
        st.info("No difficulty labels found.")
    else:
        difficulties = ["Easy", "Medium", "Hard", "unknown"]
        tabs = st.tabs(sorted(temp["problem_set"].unique()))
        for i, ps in enumerate(sorted(temp["problem_set"].unique())):
            with tabs[i]:
                sub = temp[temp["problem_set"] == ps]
                fig = go.Figure()
                for m in sorted(sub["model"].unique()):
                    row = sub[sub["model"] == m].set_index("difficulty").reindex(difficulties)["pct"].fillna(0)
                    fig.add_trace(go.Scatterpolar(
                        r=row.values.tolist(),
                        theta=difficulties,
                        fill='toself', name=m
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    title=f"Difficulty share (percent) — Problem set {ps}"
                )
                st.plotly_chart(fig, use_container_width=True)
