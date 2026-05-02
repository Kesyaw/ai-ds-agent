import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import build_graph

st.set_page_config(
    page_title="AI Data Science Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: #0a0a0f;
    }

    section[data-testid="stSidebar"] {
        background: #0f0f1a;
        border-right: 1px solid #1e1e2e;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    .hero-container {
        text-align: center;
        padding: 3rem 2rem 2rem;
    }

    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e2e8f0 0%, #a5b4fc 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 1rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #64748b;
        max-width: 500px;
        margin: 0 auto 2rem;
        line-height: 1.7;
    }

    .flow-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.3rem;
        flex-wrap: wrap;
        margin: 1.5rem 0 2.5rem;
    }

    .flow-step {
        background: #13131f;
        border: 1px solid #1e1e2e;
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        letter-spacing: 0.03em;
    }

    .flow-arrow {
        color: #6366f1;
        font-size: 0.8rem;
    }

    .metric-card {
        background: #0f0f1a;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }

    .metric-card:hover {
        border-color: #6366f1;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a5b4fc, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }

    .section-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 1rem;
    }

    .reasoning-item {
        display: flex;
        gap: 0.8rem;
        align-items: flex-start;
        padding: 0.8rem 1rem;
        background: #0f0f1a;
        border: 1px solid #1e1e2e;
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }

    .reasoning-num {
        color: #6366f1;
        font-weight: 700;
        font-size: 0.8rem;
        min-width: 20px;
    }

    .reasoning-text {
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    .progress-node {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.7rem 1rem;
        background: #0f0f1a;
        border: 1px solid #1e1e2e;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        animation: fadeIn 0.3s ease;
    }

    .progress-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #6366f1;
        box-shadow: 0 0 8px #6366f1;
        flex-shrink: 0;
    }

    .progress-text {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .report-box {
        background: #0f0f1a;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.8;
        white-space: pre-wrap;
        font-family: 'Inter', sans-serif;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.5rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    div[data-testid="stFileUploader"] {
        background: #0f0f1a !important;
        border: 1px dashed #1e1e2e !important;
        border-radius: 10px !important;
    }

    .stSelectbox > div > div {
        background: #0f0f1a !important;
        border: 1px solid #1e1e2e !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        gap: 0.5rem;
        border-bottom: 1px solid #1e1e2e;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #475569 !important;
        border: none !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
    }

    .stTabs [aria-selected="true"] {
        color: #a5b4fc !important;
        border-bottom: 2px solid #6366f1 !important;
    }

    .stDataFrame {
        background: #0f0f1a !important;
    }

    div[data-testid="stMarkdownContainer"] p {
        color: #94a3b8;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #1e1e2e, transparent);
        margin: 1.5rem 0;
    }

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0a0a0f; }
    ::-webkit-scrollbar-thumb { background: #1e1e2e; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# === SIDEBAR ===
with st.sidebar:
    st.markdown('<p class="section-title">Configuration</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Dataset", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)

        st.markdown('<p class="section-title" style="margin-top:1rem">Target Column</p>', unsafe_allow_html=True)
        target_column = st.selectbox("target", options=df_preview.columns.tolist(), label_visibility="collapsed")

        st.markdown('<p class="section-title" style="margin-top:1rem">Problem Type</p>', unsafe_allow_html=True)
        problem_type = st.selectbox("problem", options=["classification", "regression"], label_visibility="collapsed")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df_preview.shape[0]:,}</div><div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df_preview.shape[1]}</div><div class="metric-label">Cols</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("Run Agent →")
    else:
        st.markdown("""
        <div style="color: #475569; font-size: 0.8rem; line-height: 1.7; padding: 0.5rem 0;">
            Upload a CSV file to get started.<br><br>
            The agent will automatically:<br>
            • Analyze your data<br>
            • Select the best models<br>
            • Train & evaluate<br>
            • Generate a report
        </div>
        """, unsafe_allow_html=True)
        run_button = False


# === MAIN AREA ===
if not uploaded_file:
    # Hero screen
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">Powered by LangGraph</div>
        <div class="hero-title">AI Data Science<br>Agent</div>
        <div class="hero-subtitle">
            Upload any CSV dataset and let the agent reason through
            preprocessing, model selection, training, and evaluation — automatically.
        </div>
        <div class="flow-container">
            <div class="flow-step">EDA</div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">Model Selection</div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">Preprocessing</div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">Training</div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">Evaluation</div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">Report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Preview
    st.markdown('<p class="section-title">Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df_preview.head(8), use_container_width=True, hide_index=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# === RUN AGENT ===
if run_button:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    initial_state = {
        "dataset_path": temp_path,
        "target_column": target_column,
        "problem_type": problem_type,
        "df_raw": None, "df_processed": None,
        "eda_summary": None, "is_imbalanced": False,
        "missing_ratio": 0.0, "n_rows": 0, "n_features": 0,
        "has_categorical": False, "preprocessing_steps": [],
        "candidate_models": [], "model_results": {},
        "best_model": None, "needs_optimization": False,
        "iteration_count": 0, "reasoning": [],
        "report": None, "confidence_score": 0.0,
    }

    node_labels = {
        "data_understanding": "Analyzing dataset structure & patterns",
        "select_models":      "Selecting candidate models",
        "preprocessing":      "Preprocessing data adaptively",
        "modeling":           "Training models",
        "evaluation":         "Evaluating & comparing performance",
        "optimization":       "Optimizing hyperparameters",
        "report":             "Generating final report",
    }

    st.markdown('<p class="section-title">Agent Progress</p>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    log_area = st.empty()
    completed = []

    graph = build_graph()
    final_state = None
    total = len(node_labels)

    for step_output in graph.stream(initial_state):
        node_name = list(step_output.keys())[0]
        final_state = step_output[node_name]
        completed.append(node_name)

        progress_bar.progress(len(completed) / total)

        log_html = ""
        for n in completed:
            label = node_labels.get(n, n)
            log_html += f'<div class="progress-node"><div class="progress-dot"></div><div class="progress-text">{label}</div></div>'
        log_area.markdown(log_html, unsafe_allow_html=True)

    progress_bar.progress(1.0)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # === RESULTS ===
    if final_state:
        # Top metrics
        best = final_state.get("best_model", "N/A")
        conf = final_state.get("confidence_score", 0)
        iters = final_state.get("iteration_count", 0)
        n_models = len(final_state.get("model_results", {}))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{best.replace("_", " ").title()}</div><div class="metric-label">Best Model</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{conf:.0%}</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{iters}</div><div class="metric-label">Iterations</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{n_models}</div><div class="metric-label">Models Tested</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["Best Model", "All Models", "Agent Reasoning", "Report"])

        with tab1:
            results = final_state.get("model_results", {})
            if best and best in results:
                skip = {"model_object", "y_test", "y_pred", "X_test", "best_params"}
                metrics = {k: v for k, v in results[best].items() if k not in skip}
                cols = st.columns(len(metrics))
                for i, (k, v) in enumerate(metrics.items()):
                    with cols[i]:
                        val = f"{v:.4f}" if isinstance(v, float) else str(v)
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{k.upper()}</div></div>', unsafe_allow_html=True)

        with tab2:
            results = final_state.get("model_results", {})
            rows = []
            for m, r in results.items():
                skip = {"model_object", "y_test", "y_pred", "X_test"}
                row = {"Model": m}
                row.update({k: round(v, 4) if isinstance(v, float) else v
                            for k, v in r.items() if k not in skip})
                rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with tab3:
            reasoning = final_state.get("reasoning", [])
            html = ""
            for i, r in enumerate(reasoning):
                html += f'<div class="reasoning-item"><div class="reasoning-num">{i+1:02d}</div><div class="reasoning-text">{r}</div></div>'
            st.markdown(html, unsafe_allow_html=True)

        with tab4:
            report = final_state.get("report", "")
            st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)