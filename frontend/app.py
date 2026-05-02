import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import build_graph

st.set_page_config(
    page_title="AI Data Science Agent",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp { background: #fafaf9; }

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8e4f0;
}

.sidebar-logo {
    font-size: 1.3rem;
    font-weight: 700;
    color: #7c6fcd;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}

.sidebar-tagline {
    font-size: 0.75rem;
    color: #a89ec9;
    margin-bottom: 1.5rem;
}

.sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #9991b8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
    margin-top: 1rem;
}

.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #f3f0ff;
    border: 1px solid #e0d9f7;
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #7c6fcd;
    margin-right: 0.4rem;
    margin-top: 0.5rem;
}

.hero-wrap {
    padding: 3.5rem 2rem 2rem;
    text-align: center;
}

.hero-chip {
    display: inline-block;
    background: #f3f0ff;
    color: #7c6fcd;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    border: 1px solid #e0d9f7;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    color: #2d2640;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 1rem;
}

.hero-title span {
    background: linear-gradient(135deg, #a78bfa, #f9a8d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 1rem;
    color: #8b85a1;
    max-width: 460px;
    margin: 0 auto 2.5rem;
    line-height: 1.75;
    font-weight: 400;
}

.flow-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin-bottom: 3rem;
}

.flow-chip {
    background: #fff;
    border: 1px solid #e8e4f0;
    color: #6b6589;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.4rem 0.9rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(124,111,205,0.06);
}

.flow-dot {
    color: #c4b9f0;
    font-size: 0.6rem;
}

.card {
    background: #ffffff;
    border: 1px solid #ede9f8;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 4px rgba(124,111,205,0.07);
    margin-bottom: 0.8rem;
}

.metric-big {
    font-size: 2rem;
    font-weight: 700;
    color: #2d2640;
    letter-spacing: -0.02em;
    line-height: 1;
}

.metric-sub {
    font-size: 0.72rem;
    color: #a89ec9;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.35rem;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #a89ec9;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.8rem;
    margin-top: 1.5rem;
}

.progress-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1.1rem;
    background: #fff;
    border: 1px solid #ede9f8;
    border-radius: 10px;
    margin-bottom: 0.4rem;
    box-shadow: 0 1px 3px rgba(124,111,205,0.05);
}

.progress-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: linear-gradient(135deg, #a78bfa, #f9a8d4);
    flex-shrink: 0;
}

.progress-label {
    font-size: 0.84rem;
    color: #5c5578;
    font-weight: 500;
}

.reasoning-row {
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
    padding: 0.85rem 1.1rem;
    background: #fdfcff;
    border: 1px solid #ede9f8;
    border-left: 3px solid #c4b9f0;
    border-radius: 0 10px 10px 0;
    margin-bottom: 0.45rem;
}

.r-num {
    font-size: 0.75rem;
    font-weight: 700;
    color: #a78bfa;
    min-width: 22px;
    padding-top: 0.05rem;
}

.r-text {
    font-size: 0.84rem;
    color: #5c5578;
    line-height: 1.6;
}

.report-block {
    background: #fff;
    border: 1px solid #ede9f8;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    color: #5c5578;
    font-size: 0.84rem;
    line-height: 1.85;
    white-space: pre-wrap;
    box-shadow: 0 1px 4px rgba(124,111,205,0.06);
}

.stButton > button {
    background: linear-gradient(135deg, #a78bfa 0%, #f9a8d4 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(167,139,250,0.3) !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover { opacity: 0.88 !important; }

div[data-testid="stFileUploader"] {
    background: #fdfcff !important;
    border: 1.5px dashed #ddd6fe !important;
    border-radius: 10px !important;
}

.stSelectbox > div > div {
    background: #fdfcff !important;
    border: 1px solid #e0d9f7 !important;
    border-radius: 8px !important;
    color: #2d2640 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.3rem;
    border-bottom: 1px solid #ede9f8;
    padding-bottom: 0;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #a89ec9 !important;
    border: none !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.1rem !important;
    border-radius: 0 !important;
}

.stTabs [aria-selected="true"] {
    color: #7c6fcd !important;
    border-bottom: 2px solid #a78bfa !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #a78bfa, #f9a8d4) !important;
    border-radius: 999px !important;
}

.stProgress > div {
    background: #f3f0ff !important;
    border-radius: 999px !important;
}

hr { border-color: #ede9f8 !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #fafaf9; }
::-webkit-scrollbar-thumb { background: #ddd6fe; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# === SIDEBAR ===
with st.sidebar:
    st.markdown('<div class="sidebar-logo">✦ DS Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Intelligent ML pipeline</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("upload", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)

        st.markdown(
            f'<div><span class="stat-pill">⬡ {df_preview.shape[0]:,} rows</span>'
            f'<span class="stat-pill">◈ {df_preview.shape[1]} cols</span></div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="sidebar-label">Target Column</div>', unsafe_allow_html=True)
        target_column = st.selectbox("t", options=df_preview.columns.tolist(), label_visibility="collapsed")

        st.markdown('<div class="sidebar-label">Problem Type</div>', unsafe_allow_html=True)
        problem_type = st.selectbox("p", options=["classification", "regression"], label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("Run Agent →")
    else:
        st.markdown("""
        <div style="font-size:0.8rem; color:#b0a8cc; line-height:1.8; margin-top:0.5rem;">
            Upload a CSV to begin.<br><br>
            The agent will automatically<br>
            analyze, train, and report.
        </div>
        """, unsafe_allow_html=True)
        run_button = False


# === MAIN ===
if not uploaded_file:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-chip">✦ Powered by LangGraph</div>
        <div class="hero-title">Your data,<br><span>analyzed intelligently</span></div>
        <div class="hero-sub">
            Upload any CSV dataset. The agent reasons through your data,
            selects the right models, trains, evaluates, and explains every decision.
        </div>
        <div class="flow-row">
            <div class="flow-chip">EDA</div>
            <div class="flow-dot">◆</div>
            <div class="flow-chip">Model Selection</div>
            <div class="flow-dot">◆</div>
            <div class="flow-chip">Preprocessing</div>
            <div class="flow-dot">◆</div>
            <div class="flow-chip">Training</div>
            <div class="flow-dot">◆</div>
            <div class="flow-chip">Evaluation</div>
            <div class="flow-dot">◆</div>
            <div class="flow-chip">Report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown('<div class="section-label">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_preview.head(8), use_container_width=True, hide_index=True)
    st.markdown("<br>", unsafe_allow_html=True)


if run_button:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    initial_state = {
        "dataset_path": temp_path, "target_column": target_column,
        "problem_type": problem_type, "df_raw": None, "df_processed": None,
        "eda_summary": None, "is_imbalanced": False, "missing_ratio": 0.0,
        "n_rows": 0, "n_features": 0, "has_categorical": False,
        "preprocessing_steps": [], "candidate_models": [], "model_results": {},
        "best_model": None, "needs_optimization": False, "iteration_count": 0,
        "reasoning": [], "report": None, "confidence_score": 0.0,
    }

    node_labels = {
        "data_understanding": "Analyzing dataset structure & patterns",
        "select_models":      "Selecting candidate models based on data",
        "preprocessing":      "Preprocessing data adaptively",
        "modeling":           "Training candidate models",
        "evaluation":         "Evaluating & comparing performance",
        "optimization":       "Optimizing hyperparameters",
        "report":             "Generating final report",
    }

    st.markdown('<div class="section-label">Agent Progress</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    log_placeholder = st.empty()

    graph = build_graph()
    final_state = None
    completed = []
    total = len(node_labels)

    for step_output in graph.stream(initial_state):
        node_name = list(step_output.keys())[0]
        final_state = step_output[node_name]
        completed.append(node_name)
        progress_bar.progress(len(completed) / total)

        html = ""
        for n in completed:
            html += f'<div class="progress-item"><div class="progress-dot"></div><div class="progress-label">{node_labels.get(n, n)}</div></div>'
        log_placeholder.markdown(html, unsafe_allow_html=True)

    progress_bar.progress(1.0)
    st.markdown("<br>", unsafe_allow_html=True)

    if final_state:
        best = final_state.get("best_model", "N/A")
        conf = final_state.get("confidence_score", 0)
        iters = final_state.get("iteration_count", 0)
        n_models = len(final_state.get("model_results", {}))

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in zip(
            [c1, c2, c3, c4],
            [best.replace("_"," ").title(), f"{conf:.0%}", str(iters), str(n_models)],
            ["Best Model", "Confidence", "Iterations", "Models Tested"]
        ):
            with col:
                st.markdown(f'<div class="card"><div class="metric-big">{val}</div><div class="metric-sub">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["Best Model", "All Models", "Reasoning", "Report"])

        with tab1:
            results = final_state.get("model_results", {})
            if best and best in results:
                skip = {"model_object", "y_test", "y_pred", "X_test", "best_params"}
                metrics = {k: v for k, v in results[best].items() if k not in skip}
                cols = st.columns(len(metrics)) if metrics else []
                for i, (k, v) in enumerate(metrics.items()):
                    with cols[i]:
                        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                        st.markdown(f'<div class="card" style="text-align:center"><div class="metric-big">{val_str}</div><div class="metric-sub">{k.upper()}</div></div>', unsafe_allow_html=True)

        with tab2:
            rows = []
            for m, r in final_state.get("model_results", {}).items():
                skip = {"model_object", "y_test", "y_pred", "X_test"}
                row = {"Model": m}
                row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items() if k not in skip})
                rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with tab3:
            html = ""
            for i, r in enumerate(final_state.get("reasoning", [])):
                html += f'<div class="reasoning-row"><div class="r-num">{i+1:02d}</div><div class="r-text">{r}</div></div>'
            st.markdown(html, unsafe_allow_html=True)

        with tab4:
            report = final_state.get("report", "")
            st.markdown(f'<div class="report-block">{report}</div>', unsafe_allow_html=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)
