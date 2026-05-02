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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { font-family: 'Plus Jakarta Sans', sans-serif; box-sizing: border-box; }

.stApp { background: #f7f5ff; }

header[data-testid="stHeader"] {
    background: rgba(247,245,255,0.85) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #ede9f8;
}

.stMainBlockContainer { background: transparent; padding-top: 2rem; }

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #ede9f8;
    box-shadow: 4px 0 24px rgba(124,111,205,0.06);
}

/* === SIDEBAR === */
.sb-brand { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.2rem; }
.sb-icon {
    width: 30px; height: 30px; border-radius: 8px;
    background: linear-gradient(135deg, #a78bfa, #f9a8d4);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; box-shadow: 0 4px 10px rgba(167,139,250,0.35);
}
.sb-name { font-size: 1rem; font-weight: 800; color: #2d2640; letter-spacing: -0.02em; }
.sb-tag { font-size: 0.72rem; color: #b0a8cc; margin-bottom: 1.5rem; font-weight: 400; }
.sb-label {
    font-size: 0.65rem; font-weight: 700; color: #c4b9f0;
    text-transform: uppercase; letter-spacing: 0.16em;
    margin-bottom: 0.4rem; margin-top: 1.1rem;
}
.stat-row { display: flex; gap: 0.5rem; margin-top: 0.6rem; }
.stat-pill {
    flex: 1; background: #f3f0ff; border: 1px solid #e0d9f7;
    border-radius: 8px; padding: 0.5rem 0.6rem; text-align: center;
}
.stat-pill-val { font-size: 1rem; font-weight: 700; color: #7c6fcd; line-height: 1; }
.stat-pill-key { font-size: 0.6rem; color: #b0a8cc; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.15rem; }

/* === HERO === */
.hero-outer {
    min-height: 72vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 4rem 2rem 3rem; text-align: center;
    position: relative; overflow: hidden;
}
.hero-glow {
    position: absolute; width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(167,139,250,0.12) 0%, transparent 70%);
    top: 50%; left: 50%; transform: translate(-50%, -60%);
    pointer-events: none;
}
.hero-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #fff; border: 1px solid #e0d9f7;
    color: #7c6fcd; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    padding: 0.4rem 1.1rem; border-radius: 999px;
    margin-bottom: 2rem;
    box-shadow: 0 2px 12px rgba(167,139,250,0.15);
}
.hero-chip-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: linear-gradient(135deg, #a78bfa, #f9a8d4);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.7); }
}
.hero-title {
    font-size: 3.8rem; font-weight: 800; color: #1e1a33;
    letter-spacing: -0.04em; line-height: 1.1; margin-bottom: 1.2rem;
}
.hero-title .grad {
    background: linear-gradient(135deg, #a78bfa 0%, #f472b6 50%, #fb923c 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem; color: #8b85a1; max-width: 480px;
    margin: 0 auto 3rem; line-height: 1.8; font-weight: 400;
}
.flow-row {
    display: flex; align-items: center; justify-content: center;
    gap: 0.35rem; flex-wrap: wrap;
}
.flow-chip {
    background: #fff; border: 1px solid #ede9f8; color: #6b6589;
    font-size: 0.7rem; font-weight: 600; padding: 0.45rem 1rem;
    border-radius: 8px; box-shadow: 0 1px 4px rgba(124,111,205,0.08);
    transition: all 0.2s;
}
.flow-chip:hover { border-color: #c4b9f0; color: #7c6fcd; }
.flow-arrow { color: #ddd6fe; font-size: 0.55rem; }

/* === CARDS === */
.card {
    background: #fff; border: 1px solid #ede9f8; border-radius: 16px;
    padding: 1.4rem 1.6rem; box-shadow: 0 1px 8px rgba(124,111,205,0.07);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.card:hover { box-shadow: 0 4px 20px rgba(124,111,205,0.12); border-color: #d8d0f5; }
.metric-big {
    font-size: 1.9rem; font-weight: 800; color: #2d2640;
    letter-spacing: -0.03em; line-height: 1;
}
.metric-sub {
    font-size: 0.65rem; font-weight: 600; color: #c4b9f0;
    text-transform: uppercase; letter-spacing: 0.14em; margin-top: 0.4rem;
}

/* === SECTION LABEL === */
.section-label {
    font-size: 0.65rem; font-weight: 700; color: #c4b9f0;
    text-transform: uppercase; letter-spacing: 0.18em;
    margin-bottom: 0.9rem; margin-top: 1.8rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after {
    content: ''; flex: 1; height: 1px; background: #ede9f8;
}

/* === PROGRESS === */
.progress-item {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.8rem 1.1rem; background: #fff;
    border: 1px solid #ede9f8; border-radius: 10px;
    margin-bottom: 0.4rem;
    box-shadow: 0 1px 4px rgba(124,111,205,0.05);
    animation: slideIn 0.25s ease forwards;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.progress-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient(135deg, #a78bfa, #f9a8d4);
    box-shadow: 0 0 6px rgba(167,139,250,0.5);
}
.progress-label { font-size: 0.84rem; color: #5c5578; font-weight: 500; }
.progress-check { margin-left: auto; color: #a78bfa; font-size: 0.9rem; }

/* === REASONING === */
.reasoning-row {
    display: flex; gap: 0.9rem; align-items: flex-start;
    padding: 0.85rem 1.1rem; background: #fdfcff;
    border: 1px solid #ede9f8; border-left: 3px solid #ddd6fe;
    border-radius: 0 10px 10px 0; margin-bottom: 0.4rem;
    transition: border-left-color 0.2s;
}
.reasoning-row:hover { border-left-color: #a78bfa; }
.r-num { font-size: 0.72rem; font-weight: 700; color: #c4b9f0; min-width: 22px; padding-top: 0.1rem; font-family: 'DM Mono', monospace; }
.r-text { font-size: 0.83rem; color: #5c5578; line-height: 1.65; }

/* === BUTTONS === */
.stButton > button {
    background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 0.88rem !important; padding: 0.65rem 1.5rem !important;
    width: 100% !important; letter-spacing: 0.01em !important;
    box-shadow: 0 4px 18px rgba(167,139,250,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(167,139,250,0.45) !important;
}

/* === INPUTS === */
div[data-testid="stFileUploader"] {
    background: #fdfcff !important;
    border: 1.5px dashed #ddd6fe !important;
    border-radius: 10px !important;
}
.stSelectbox > div > div {
    background: #fdfcff !important; border: 1px solid #e0d9f7 !important;
    border-radius: 8px !important; color: #2d2640 !important;
}

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important; gap: 0.2rem;
    border-bottom: 1px solid #ede9f8;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #b0a8cc !important;
    border: none !important; font-size: 0.83rem !important;
    font-weight: 600 !important; padding: 0.55rem 1.1rem !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #7c6fcd !important;
    border-bottom: 2px solid #a78bfa !important;
}

/* === PROGRESS BAR === */
.stProgress > div > div {
    background: linear-gradient(90deg, #a78bfa, #f472b6) !important;
    border-radius: 999px !important;
}
.stProgress > div { background: #f3f0ff !important; border-radius: 999px !important; }

/* === DATAFRAME === */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f7f5ff; }
::-webkit-scrollbar-thumb { background: #ddd6fe; border-radius: 3px; }

hr { border-color: #ede9f8 !important; }
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-icon">✦</div>
        <div class="sb-name">DS Agent</div>
    </div>
    <div class="sb-tag">Intelligent ML pipeline</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("upload", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill">
                <div class="stat-pill-val">{df_preview.shape[0]:,}</div>
                <div class="stat-pill-key">Rows</div>
            </div>
            <div class="stat-pill">
                <div class="stat-pill-val">{df_preview.shape[1]}</div>
                <div class="stat-pill-key">Cols</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Target Column</div>', unsafe_allow_html=True)
        target_column = st.selectbox("t", options=df_preview.columns.tolist(), label_visibility="collapsed")

        st.markdown('<div class="sb-label">Problem Type</div>', unsafe_allow_html=True)
        problem_type = st.selectbox("p", options=["classification", "regression"], label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("✦ Run Agent")
    else:
        st.markdown("""
        <div style="font-size:0.8rem; color:#c4b9f0; line-height:1.9; margin-top:0.8rem;
                    background:#fdfcff; border:1px solid #ede9f8; border-radius:10px; padding:1rem;">
            Upload a CSV to begin.<br><br>
            The agent will automatically<br>
            · Analyze your data<br>
            · Select the best models<br>
            · Train & evaluate<br>
            · Generate a report
        </div>
        """, unsafe_allow_html=True)
        run_button = False


# ─── HERO ───────────────────────────────────────────────────
if not uploaded_file:
    st.markdown("""
    <div class="hero-outer">
        <div class="hero-glow"></div>
        <div class="hero-chip"><div class="hero-chip-dot"></div>Powered by LangGraph</div>
        <div class="hero-title">Your data,<br><span class="grad">analyzed intelligently</span></div>
        <div class="hero-sub">
            Upload any CSV dataset. The agent reasons through your data,
            selects the right models, trains, evaluates, and explains every decision.
        </div>
        <div class="flow-row">
            <div class="flow-chip">EDA</div>
            <div class="flow-arrow">▶</div>
            <div class="flow-chip">Model Selection</div>
            <div class="flow-arrow">▶</div>
            <div class="flow-chip">Preprocessing</div>
            <div class="flow-arrow">▶</div>
            <div class="flow-chip">Training</div>
            <div class="flow-arrow">▶</div>
            <div class="flow-chip">Evaluation</div>
            <div class="flow-arrow">▶</div>
            <div class="flow-chip">Report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown('<div class="section-label">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_preview.head(8), use_container_width=True, hide_index=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ─── RUN AGENT ──────────────────────────────────────────────
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
            html += f'''
            <div class="progress-item">
                <div class="progress-dot"></div>
                <div class="progress-label">{node_labels.get(n, n)}</div>
                <div class="progress-check">✓</div>
            </div>'''
        log_placeholder.markdown(html, unsafe_allow_html=True)

    progress_bar.progress(1.0)
    st.markdown("<br>", unsafe_allow_html=True)

    if final_state:
        best = final_state.get("best_model", "N/A")
        conf = final_state.get("confidence_score", 0)
        iters = final_state.get("iteration_count", 0)
        n_models = len(final_state.get("model_results", {}))

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label, color in zip(
            [c1, c2, c3, c4],
            [best.replace("_"," ").title(), f"{conf:.0%}", str(iters), str(n_models)],
            ["Best Model", "Confidence", "Iterations", "Models Tested"],
            ["#a78bfa", "#f472b6", "#fb923c", "#34d399"]
        ):
            with col:
                st.markdown(f'''
                <div class="card">
                    <div class="metric-big" style="color:{color}">{val}</div>
                    <div class="metric-sub">{label}</div>
                </div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["✦ Best Model", "⊞ All Models", "◎ Reasoning", "✐ Report"])

        with tab1:
            results = final_state.get("model_results", {})
            if best and best in results:
                skip = {"model_object", "y_test", "y_pred", "X_test", "best_params"}
                metrics = {k: v for k, v in results[best].items() if k not in skip}
                if metrics:
                    cols = st.columns(len(metrics))
                    metric_colors = ["#a78bfa", "#f472b6", "#fb923c", "#34d399", "#38bdf8"]
                    for i, (k, v) in enumerate(metrics.items()):
                        with cols[i]:
                            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                            color = metric_colors[i % len(metric_colors)]
                            st.markdown(f'''
                            <div class="card" style="text-align:center">
                                <div class="metric-big" style="color:{color}">{val_str}</div>
                                <div class="metric-sub">{k.upper()}</div>
                            </div>''', unsafe_allow_html=True)

        with tab2:
            rows = []
            for m, r in final_state.get("model_results", {}).items():
                skip = {"model_object", "y_test", "y_pred", "X_test"}
                row = {"Model": m.replace("_"," ").title()}
                row.update({k.upper(): round(v, 4) if isinstance(v, float) else v
                            for k, v in r.items() if k not in skip})
                rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with tab3:
            html = ""
            for i, r in enumerate(final_state.get("reasoning", [])):
                html += f'''
                <div class="reasoning-row">
                    <div class="r-num">{i+1:02d}</div>
                    <div class="r-text">{r}</div>
                </div>'''
            st.markdown(html, unsafe_allow_html=True)

        with tab4:
            report = final_state.get("report", "")
            if report:
                st.markdown(report, unsafe_allow_html=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)
