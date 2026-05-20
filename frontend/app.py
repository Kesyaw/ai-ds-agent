import streamlit as st
import pandas as pd
import os
import sys
import requests
import io

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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after {
    font-family: 'Plus Jakarta Sans', sans-serif;
    box-sizing: border-box;
}

.stApp { background: #f7f5ff; }

header[data-testid="stHeader"] {
    background: rgba(247,245,255,0.9) !important;
    backdrop-filter: blur(16px);
    border-bottom: 1px solid #ede9f8;
}

.stMainBlockContainer {
    background: transparent;
    padding-top: 1.5rem;
    max-width: 1100px;
}

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #ede9f8;
    box-shadow: 4px 0 30px rgba(124,111,205,0.07);
}

/* ── SIDEBAR ── */
.sb-brand {
    display: flex; align-items: center; gap: 0.55rem;
    margin-bottom: 0.25rem;
}
.sb-icon {
    width: 32px; height: 32px; border-radius: 9px;
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem;
    box-shadow: 0 4px 12px rgba(167,139,250,0.4);
}
.sb-name {
    font-size: 1.05rem; font-weight: 800;
    color: #1e1a33; letter-spacing: -0.02em;
}
.sb-tag {
    font-size: 0.71rem; color: #b0a8cc;
    margin-bottom: 1.6rem; font-weight: 400;
    letter-spacing: 0.01em;
}
.sb-label {
    font-size: 0.62rem; font-weight: 700; color: #c4b9f0;
    text-transform: uppercase; letter-spacing: 0.18em;
    margin-bottom: 0.45rem; margin-top: 1.2rem;
}
.stat-row { display: flex; gap: 0.5rem; margin-top: 0.65rem; }
.stat-pill {
    flex: 1; background: #f3f0ff; border: 1px solid #e0d9f7;
    border-radius: 9px; padding: 0.55rem 0.5rem; text-align: center;
}
.stat-pill-val {
    font-size: 1.05rem; font-weight: 800; color: #7c6fcd; line-height: 1;
}
.stat-pill-key {
    font-size: 0.58rem; color: #b0a8cc; text-transform: uppercase;
    letter-spacing: 0.12em; margin-top: 0.18rem;
}
.sb-hint {
    font-size: 0.78rem; color: #c4b9f0; line-height: 1.85;
    margin-top: 0.6rem; background: #fdfcff;
    border: 1px solid #ede9f8; border-radius: 10px;
    padding: 0.9rem 1rem;
}

/* ── SAMPLE CARDS ── */
.sample-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem; margin-bottom: 2rem;
}
.sample-card {
    background: #fff; border: 1.5px solid #ede9f8;
    border-radius: 14px; padding: 1.1rem 1.2rem;
    cursor: pointer; transition: all 0.2s;
    box-shadow: 0 1px 6px rgba(124,111,205,0.06);
    text-decoration: none; display: block;
}
.sample-card:hover {
    border-color: #c4b9f0;
    box-shadow: 0 6px 24px rgba(124,111,205,0.13);
    transform: translateY(-2px);
}
.sample-icon {
    font-size: 1.5rem; margin-bottom: 0.6rem;
}
.sample-title {
    font-size: 0.88rem; font-weight: 700; color: #2d2640;
    margin-bottom: 0.25rem;
}
.sample-desc {
    font-size: 0.74rem; color: #a89ec9; line-height: 1.55;
    margin-bottom: 0.6rem;
}
.sample-tags { display: flex; gap: 0.35rem; flex-wrap: wrap; }
.sample-tag {
    background: #f3f0ff; border: 1px solid #e0d9f7;
    color: #7c6fcd; font-size: 0.62rem; font-weight: 600;
    padding: 0.18rem 0.55rem; border-radius: 999px;
}
.sample-tag.reg {
    background: #fdf4ff; border-color: #f0d9f7; color: #a855f7;
}

/* ── HERO ── */
.hero-outer {
    min-height: 70vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 3rem 1rem 1rem; text-align: center;
    position: relative;
}
.hero-glow-1 {
    position: absolute; width: 600px; height: 400px;
    background: radial-gradient(ellipse, rgba(167,139,250,0.1) 0%, transparent 70%);
    top: 10%; left: 50%; transform: translateX(-50%);
    pointer-events: none;
}
.hero-glow-2 {
    position: absolute; width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(249,168,212,0.08) 0%, transparent 70%);
    top: 40%; right: 5%;
    pointer-events: none;
}
.hero-chip {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: #fff; border: 1px solid #e0d9f7;
    color: #7c6fcd; font-size: 0.67rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
    padding: 0.4rem 1.1rem; border-radius: 999px;
    margin-bottom: 2rem;
    box-shadow: 0 2px 14px rgba(167,139,250,0.18);
}
.hero-chip-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.6); }
}
.hero-title {
    font-size: 4rem; font-weight: 800; color: #1e1a33;
    letter-spacing: -0.04em; line-height: 1.08;
    margin-bottom: 1.3rem;
}
.hero-title .grad {
    background: linear-gradient(135deg, #a78bfa 0%, #f472b6 55%, #fb923c 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem; color: #8b85a1; max-width: 500px;
    margin: 0 auto 3rem; line-height: 1.8; font-weight: 400;
}
.flow-row {
    display: flex; align-items: center; justify-content: center;
    gap: 0.3rem; flex-wrap: wrap; margin-bottom: 0.5rem;
}
.flow-chip {
    background: #fff; border: 1px solid #ede9f8; color: #6b6589;
    font-size: 0.69rem; font-weight: 600; padding: 0.45rem 1rem;
    border-radius: 8px; box-shadow: 0 1px 4px rgba(124,111,205,0.07);
}
.flow-arrow { color: #ddd6fe; font-size: 0.5rem; }

.try-label {
    font-size: 0.68rem; font-weight: 700; color: #c4b9f0;
    text-transform: uppercase; letter-spacing: 0.18em;
    margin: 2.5rem 0 1rem; text-align: center;
}

/* ── SECTION LABEL ── */
.section-label {
    font-size: 0.62rem; font-weight: 700; color: #c4b9f0;
    text-transform: uppercase; letter-spacing: 0.2em;
    margin-bottom: 0.9rem; margin-top: 1.8rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #ede9f8, transparent);
}

/* ── CARDS ── */
.card {
    background: #fff; border: 1px solid #ede9f8;
    border-radius: 16px; padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 8px rgba(124,111,205,0.07);
    transition: all 0.2s;
}
.card:hover {
    box-shadow: 0 6px 24px rgba(124,111,205,0.12);
    border-color: #d8d0f5;
}
.metric-big {
    font-size: 1.85rem; font-weight: 800; color: #2d2640;
    letter-spacing: -0.03em; line-height: 1;
}
.metric-sub {
    font-size: 0.63rem; font-weight: 700; color: #c4b9f0;
    text-transform: uppercase; letter-spacing: 0.16em;
    margin-top: 0.45rem;
}

/* ── PROGRESS ── */
.progress-item {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.8rem 1.1rem; background: #fff;
    border: 1px solid #ede9f8; border-radius: 10px;
    margin-bottom: 0.38rem;
    box-shadow: 0 1px 4px rgba(124,111,205,0.05);
    animation: slideUp 0.2s ease forwards;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}
.progress-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    box-shadow: 0 0 8px rgba(167,139,250,0.5);
}
.progress-label { font-size: 0.83rem; color: #5c5578; font-weight: 500; }
.progress-check { margin-left: auto; color: #a78bfa; font-size: 0.85rem; font-weight: 700; }

/* ── REASONING ── */
.reasoning-row {
    display: flex; gap: 0.85rem; align-items: flex-start;
    padding: 0.8rem 1.05rem; background: #fdfcff;
    border: 1px solid #ede9f8; border-left: 3px solid #ddd6fe;
    border-radius: 0 10px 10px 0; margin-bottom: 0.38rem;
    transition: border-left-color 0.2s;
}
.reasoning-row:hover { border-left-color: #a78bfa; }
.r-num {
    font-size: 0.7rem; font-weight: 700; color: #c4b9f0;
    min-width: 22px; padding-top: 0.1rem;
    font-family: 'DM Mono', monospace;
}
.r-text { font-size: 0.82rem; color: #5c5578; line-height: 1.65; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 0.87rem !important; width: 100% !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 18px rgba(167,139,250,0.35) !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"][key="back_btn"] > button {
    background: #fff !important;
    color: #a89ec9 !important;
    border: 1px solid #ede9f8 !important;
    box-shadow: none !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    width: auto !important;
    padding: 0.4rem 1rem !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 7px 26px rgba(167,139,250,0.45) !important;
}
.stDownloadButton > button {
    background: #fff !important; color: #7c6fcd !important;
    border: 1.5px solid #e0d9f7 !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.83rem !important; width: 100% !important;
    box-shadow: 0 1px 6px rgba(124,111,205,0.08) !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    border-color: #a78bfa !important;
    box-shadow: 0 4px 16px rgba(167,139,250,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── INPUTS ── */
div[data-testid="stFileUploader"] {
    background: #fdfcff !important;
    border: 1.5px dashed #ddd6fe !important;
    border-radius: 10px !important;
}
.stSelectbox > div > div {
    background: #fdfcff !important; border: 1px solid #e0d9f7 !important;
    border-radius: 8px !important; color: #2d2640 !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important; gap: 0.1rem;
    border-bottom: 1px solid #ede9f8;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #b0a8cc !important;
    border: none !important; font-size: 0.82rem !important;
    font-weight: 600 !important; padding: 0.55rem 1.1rem !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #7c6fcd !important;
    border-bottom: 2px solid #a78bfa !important;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #a78bfa, #f472b6) !important;
    border-radius: 999px !important;
}
.stProgress > div {
    background: #ede9f8 !important; border-radius: 999px !important;
    height: 6px !important;
}

/* ── DATAFRAME ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }
[data-testid="stDataFrame"] > div {
    border-radius: 12px !important;
    border: 1px solid #ede9f8 !important;
}

/* ── DIVIDER ── */
hr { border-color: #ede9f8 !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #f7f5ff; }
::-webkit-scrollbar-thumb { background: #ddd6fe; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #c4b9f0; }
</style>
""", unsafe_allow_html=True)

# ── SAMPLE DATASETS ──────────────────────────────────────────
SAMPLES = {
    "titanic": {
        "label": "Titanic Survival",
        "icon": "🚢",
        "desc": "Predict passenger survival. Has missing values & categorical features.",
        "target": "Survived",
        "type": "classification",
        "tags": ["Classification", "Missing Values", "891 rows"],
        "tag_class": "",
        "url": "https://raw.githubusercontent.com/Kesyaw/ai-ds-agent/main/samples/customer_churn_classification.csv",
    },
    "churn": {
        "label": "Customer Churn",
        "icon": "📉",
        "desc": "Predict which customers will churn. Balanced classes.",
        "target": "churn",
        "type": "classification",
        "tags": ["Classification", "Balanced", "800 rows"],
        "tag_class": "",
        "url": "https://raw.githubusercontent.com/Kesyaw/ai-ds-agent/main/samples/customer_churn_classification.csv",
    },
    "house": {
        "label": "House Prices",
        "icon": "🏡",
        "desc": "Predict house prices from area, location & features.",
        "target": "price",
        "type": "regression",
        "tags": ["Regression", "Mixed Features", "500 rows"],
        "tag_class": "reg",
        "url": "https://raw.githubusercontent.com/Kesyaw/ai-ds-agent/main/samples/house_prices_regression.csv",
    },
}


def load_sample(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url)
    except:
        return None


# ── SESSION STATE ─────────────────────────────────────────────
if "df_preview" not in st.session_state:
    st.session_state.df_preview = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = "classification"
if "source_label" not in st.session_state:
    st.session_state.source_label = None
if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-icon">✦</div>
        <div class="sb-name">DS Agent</div>
    </div>
    <div class="sb-tag">Intelligent ML pipeline · LangGraph</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Upload Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("upload", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)

        # Simpan ke temp file
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.df_preview = df
        st.session_state.uploaded_path = temp_path
        st.session_state.source_label = uploaded_file.name

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill">
                <div class="stat-pill-val">{df.shape[0]:,}</div>
                <div class="stat-pill-key">Rows</div>
            </div>
            <div class="stat-pill">
                <div class="stat-pill-val">{df.shape[1]}</div>
                <div class="stat-pill-key">Cols</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Target Column</div>', unsafe_allow_html=True)
        st.session_state.target_column = st.selectbox(
            "t", options=df.columns.tolist(), label_visibility="collapsed"
        )
        st.markdown('<div class="sb-label">Problem Type</div>', unsafe_allow_html=True)
        st.session_state.problem_type = st.selectbox(
            "p", options=["classification", "regression"], label_visibility="collapsed"
        )

    else:
        st.markdown("""
        <div class="sb-hint">
            Upload a CSV or try a sample dataset below.<br><br>
            The agent will automatically<br>
            · Analyze your data<br>
            · Select the best models<br>
            · Train & evaluate<br>
            · Explain every decision<br>
            · Generate a full report
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.df_preview is not None:
        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.source_label:
            st.markdown(
                f'<div style="font-size:0.72rem;color:#a89ec9;margin-bottom:0.5rem;">📂 {st.session_state.source_label}</div>',
                unsafe_allow_html=True
            )

        run_button = st.button("✦ Run Agent")
    else:
        run_button = False


# ── MAIN ──────────────────────────────────────────────────────
if st.session_state.df_preview is None:
    # Hero
    st.markdown("""
    <div class="hero-outer">
        <div class="hero-glow-1"></div>
        <div class="hero-glow-2"></div>
        <div class="hero-chip">
            <div class="hero-chip-dot"></div>
            Powered by LangGraph
        </div>
        <div class="hero-title">
            Your data,<br>
            <span class="grad">analyzed intelligently</span>
        </div>
        <div class="hero-sub">
            Upload any CSV dataset. The agent reasons through your data,
            selects the right models, trains, evaluates, and explains
            every single decision it makes.
        </div>
        <div class="flow-row">
            <div class="flow-chip">EDA</div>
            <div class="flow-arrow">▸</div>
            <div class="flow-chip">Model Selection</div>
            <div class="flow-arrow">▸</div>
            <div class="flow-chip">Preprocessing</div>
            <div class="flow-arrow">▸</div>
            <div class="flow-chip">Training</div>
            <div class="flow-arrow">▸</div>
            <div class="flow-chip">Evaluation</div>
            <div class="flow-arrow">▸</div>
            <div class="flow-chip">Report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample dataset section
    st.markdown('<div class="try-label">✦ or try a sample dataset</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    sample_keys = list(SAMPLES.keys())

    for i, key in enumerate(sample_keys):
        s = SAMPLES[key]
        with cols[i]:
            tags_html = "".join(
                f'<span class="sample-tag {s["tag_class"]}">{t}</span>'
                for t in s["tags"]
            )
            st.markdown(f"""
            <div class="sample-card">
                <div class="sample-icon">{s["icon"]}</div>
                <div class="sample-title">{s["label"]}</div>
                <div class="sample-desc">{s["desc"]}</div>
                <div class="sample-tags">{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Try {s['label']}", key=f"btn_{key}", use_container_width=True):
                with st.spinner(f"Loading {s['label']}..."):
                    df = load_sample(s["url"])
                if df is not None:
                    temp_path = f"temp_sample_{key}.csv"
                    df.to_csv(temp_path, index=False)
                    st.session_state.df_preview = df
                    st.session_state.uploaded_path = temp_path
                    st.session_state.target_column = s["target"]
                    st.session_state.problem_type = s["type"]
                    st.session_state.source_label = f"{s['icon']} {s['label']} (sample)"
                    st.rerun()
                else:
                    st.error("Failed to load sample")

else:
    # Dataset loaded — show preview
    df_preview = st.session_state.df_preview

    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("← Back", key="back_btn"):
            st.session_state.df_preview = None
            st.session_state.uploaded_path = None
            st.session_state.target_column = None
            st.session_state.source_label = None
            st.rerun()

    st.markdown('<div class="section-label">Dataset Preview</div>', unsafe_allow_html=True)

    # Info bar
    target = st.session_state.target_column or ""
    ptype = st.session_state.problem_type or ""
    source = st.session_state.source_label or ""

    st.markdown(f"""
    <div style="display:flex;gap:0.6rem;margin-bottom:1rem;flex-wrap:wrap;align-items:center;">
        <div style="background:#f3f0ff;border:1px solid #e0d9f7;border-radius:8px;
                    padding:0.35rem 0.9rem;font-size:0.76rem;font-weight:600;color:#7c6fcd;">
            📂 {source}
        </div>
        <div style="background:#fdf4ff;border:1px solid #f0d9f7;border-radius:8px;
                    padding:0.35rem 0.9rem;font-size:0.76rem;font-weight:600;color:#a855f7;">
            🎯 Target: {target}
        </div>
        <div style="background:#f0fdf4;border:1px solid #d1fae5;border-radius:8px;
                    padding:0.35rem 0.9rem;font-size:0.76rem;font-weight:600;color:#10b981;">
            ⚙ {ptype.title()}
        </div>
        <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;
                    padding:0.35rem 0.9rem;font-size:0.76rem;font-weight:600;color:#f97316;">
            ⬡ {df_preview.shape[0]:,} × {df_preview.shape[1]} cols
        </div>
        <div style="margin-left:auto;">
            <span style="font-size:0.72rem;color:#c4b9f0;cursor:pointer;"
                  onclick="window.location.reload()">← use different dataset</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df_preview.head(8), use_container_width=True, hide_index=True)

    # Config jika dari sample (sidebar mungkin tidak di-set)
    if not uploaded_file:
        st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div style="font-size:0.7rem;font-weight:700;color:#c4b9f0;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:0.3rem;">Target Column</div>', unsafe_allow_html=True)
            st.session_state.target_column = st.selectbox(
                "tc", options=df_preview.columns.tolist(),
                index=df_preview.columns.tolist().index(st.session_state.target_column)
                    if st.session_state.target_column in df_preview.columns else 0,
                label_visibility="collapsed"
            )
        with c2:
            st.markdown('<div style="font-size:0.7rem;font-weight:700;color:#c4b9f0;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:0.3rem;">Problem Type</div>', unsafe_allow_html=True)
            st.session_state.problem_type = st.selectbox(
                "pt", options=["classification", "regression"],
                index=0 if st.session_state.problem_type == "classification" else 1,
                label_visibility="collapsed"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("✦ Run Agent", use_container_width=True)


# ── RUN AGENT ─────────────────────────────────────────────────
if run_button and st.session_state.df_preview is not None:

    dataset_path = st.session_state.uploaded_path
    target_column = st.session_state.target_column
    problem_type = st.session_state.problem_type

    if not dataset_path or not os.path.exists(dataset_path):
        st.error("Dataset file not found. Please re-upload.")
        st.stop()

    initial_state = {
        "dataset_path": dataset_path,
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
            html += f"""
            <div class="progress-item">
                <div class="progress-dot"></div>
                <div class="progress-label">{node_labels.get(n, n)}</div>
                <div class="progress-check">✓</div>
            </div>"""
        log_placeholder.markdown(html, unsafe_allow_html=True)

    progress_bar.progress(1.0)
    st.markdown("<br>", unsafe_allow_html=True)

    if final_state:
        best = final_state.get("best_model", "N/A")
        conf = final_state.get("confidence_score", 0)
        iters = final_state.get("iteration_count", 0)
        n_models = len(final_state.get("model_results", {}))

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        for col, val, label, color in zip(
            [c1, c2, c3, c4],
            [best.replace("_", " ").title(), f"{conf:.0%}", str(iters), str(n_models)],
            ["Best Model", "Confidence", "Iterations", "Models Tested"],
            ["#a78bfa", "#f472b6", "#fb923c", "#34d399"]
        ):
            with col:
                st.markdown(f"""
                <div class="card">
                    <div class="metric-big" style="color:{color}">{val}</div>
                    <div class="metric-sub">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Export buttons
        from tools.export_tools import export_model_pkl, export_model_metadata
        st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
        ex1, ex2 = st.columns(2)
        with ex1:
            model_bytes = export_model_pkl(final_state)
            if model_bytes:
                st.download_button(
                    "⬇ Download Model (.pkl)",
                    data=model_bytes,
                    file_name=f"{best}_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        with ex2:
            meta_json = export_model_metadata(final_state)
            if meta_json:
                st.download_button(
                    "⬇ Download Metadata (.json)",
                    data=meta_json,
                    file_name=f"{best}_metadata.json",
                    mime="application/json",
                    use_container_width=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "✦ Best Model", "⊞ All Models", "◎ Reasoning", "✐ Report"
        ])

        with tab1:
            results = final_state.get("model_results", {})
            if best and best in results:
                skip = {"model_object", "y_test", "y_pred", "X_test", "best_params"}
                metrics = {k: v for k, v in results[best].items() if k not in skip}
                if metrics:
                    metric_colors = ["#a78bfa", "#f472b6", "#fb923c", "#34d399", "#38bdf8"]
                    cols = st.columns(len(metrics))
                    for i, (k, v) in enumerate(metrics.items()):
                        with cols[i]:
                            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                            st.markdown(f"""
                            <div class="card" style="text-align:center">
                                <div class="metric-big" style="color:{metric_colors[i % len(metric_colors)]}">{val_str}</div>
                                <div class="metric-sub">{k.upper()}</div>
                            </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Feature Importance (SHAP)</div>', unsafe_allow_html=True)

                model_obj = results[best].get("model_object")
                X_test_data = results[best].get("X_test")

                if model_obj is not None and X_test_data is not None:
                    from tools.shap_tools import get_shap_explanation
                    with st.spinner("Generating SHAP explanation..."):
                        shap_result = get_shap_explanation(
                            model_obj, X_test_data, best,
                            final_state.get("problem_type", "classification")
                        )
                    if shap_result["status"] == "success" and shap_result["image_b64"]:
                        st.markdown(f"""
                        <div class="card" style="padding:1.5rem">
                            <img src="data:image/png;base64,{shap_result['image_b64']}"
                                 style="width:100%;border-radius:8px;" />
                        </div>""", unsafe_allow_html=True)

                        top5 = list(shap_result["importance_dict"].items())[:5]
                        pills = '<div style="display:flex;gap:0.45rem;flex-wrap:wrap;margin-top:1rem;">'
                        for feat, val in top5:
                            pills += f'<div style="background:#f3f0ff;border:1px solid #e0d9f7;border-radius:999px;padding:0.3rem 0.85rem;font-size:0.73rem;font-weight:600;color:#7c6fcd;">{feat} <span style="color:#c4b9f0;font-weight:400;">{val:.3f}</span></div>'
                        pills += "</div>"
                        st.markdown(pills, unsafe_allow_html=True)
                    else:
                        st.info(f"SHAP not available: {shap_result['status']}")

        with tab2:
            rows = []
            for m, r in final_state.get("model_results", {}).items():
                skip = {"model_object", "y_test", "y_pred", "X_test"}
                row = {"Model": m.replace("_", " ").title()}
                row.update({
                    k.upper(): round(v, 4) if isinstance(v, float) else v
                    for k, v in r.items() if k not in skip
                })
                rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with tab3:
            html = ""
            for i, r in enumerate(final_state.get("reasoning", [])):
                html += f"""
                <div class="reasoning-row">
                    <div class="r-num">{i+1:02d}</div>
                    <div class="r-text">{r}</div>
                </div>"""
            st.markdown(html, unsafe_allow_html=True)

        with tab4:
            report = final_state.get("report", "")
            if report:
                st.markdown(report, unsafe_allow_html=True)

    # Cleanup
    if dataset_path and os.path.exists(dataset_path):
        os.remove(dataset_path)
        st.session_state.uploaded_path = None
