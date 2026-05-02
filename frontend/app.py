import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import build_graph

st.set_page_config(
    page_title="AI Data Science Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Data Science Agent")
st.markdown("Upload dataset CSV, tentukan target kolom, dan biarkan agent bekerja otomatis.")

# === Sidebar: Input ===
with st.sidebar:
    st.header("⚙️ Konfigurasi")

    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)

        target_column = st.selectbox(
            "Target Column",
            options=df_preview.columns.tolist()
        )

        problem_type = st.selectbox(
            "Problem Type",
            options=["classification", "regression"]
        )

        st.markdown("---")
        st.markdown(f"**Rows:** {df_preview.shape[0]}")
        st.markdown(f"**Columns:** {df_preview.shape[1]}")

        run_button = st.button("🚀 Jalankan Agent", type="primary", use_container_width=True)
    else:
        st.info("Upload CSV untuk memulai")
        run_button = False

# === Main area ===
if uploaded_file:
    st.subheader("👀 Preview Dataset")
    st.dataframe(df_preview.head(10), use_container_width=True)

if run_button:
    # Simpan file ke temp
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    initial_state = {
        "dataset_path": temp_path,
        "target_column": target_column,
        "problem_type": problem_type,
        "df_raw": None,
        "df_processed": None,
        "eda_summary": None,
        "is_imbalanced": False,
        "missing_ratio": 0.0,
        "n_rows": 0,
        "n_features": 0,
        "has_categorical": False,
        "preprocessing_steps": [],
        "candidate_models": [],
        "model_results": {},
        "best_model": None,
        "needs_optimization": False,
        "iteration_count": 0,
        "reasoning": [],
        "report": None,
        "confidence_score": 0.0,
    }

    # === Progress tracking ===
    st.markdown("---")
    st.subheader("⚡ Agent Progress")

    progress_bar = st.progress(0)
    status_container = st.empty()
    log_container = st.container()

    node_steps = {
        "data_understanding": (1, "📊 Analisis dataset..."),
        "select_models":      (2, "🧠 Memilih kandidat model..."),
        "preprocessing":      (3, "🔧 Preprocessing data..."),
        "modeling":           (4, "🏋️ Training model..."),
        "evaluation":         (5, "📈 Evaluasi performa..."),
        "optimization":       (6, "⚡ Optimasi hyperparameter..."),
        "report":             (7, "📝 Membuat laporan..."),
    }
    total_steps = 7

    graph = build_graph()
    final_state = None

    with st.spinner("Agent sedang berjalan..."):
        for step_output in graph.stream(initial_state):
            node_name = list(step_output.keys())[0]
            final_state = step_output[node_name]

            step_num, step_label = node_steps.get(node_name, (1, node_name))
            progress = step_num / total_steps
            progress_bar.progress(progress)
            status_container.info(f"Step {step_num}/{total_steps}: {step_label}")

            with log_container:
                st.success(f"✅ {node_name} selesai")

    progress_bar.progress(1.0)
    status_container.success("🎉 Agent selesai!")

    # === Tampilkan hasil ===
    if final_state:
        st.markdown("---")

        # Tabs untuk hasil
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Best Model",
            "📊 Semua Model",
            "🧠 Agent Reasoning",
            "📝 Laporan"
        ])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Model", final_state.get("best_model", "N/A"))
            with col2:
                conf = final_state.get("confidence_score", 0)
                st.metric("Confidence Score", f"{conf:.0%}")
            with col3:
                st.metric("Total Iterasi", final_state.get("iteration_count", 0))

            st.markdown("#### Metrics Best Model")
            best = final_state.get("best_model")
            results = final_state.get("model_results", {})
            if best and best in results:
                skip = {"model_object", "y_test", "y_pred", "X_test", "best_params"}
                metrics = {k: v for k, v in results[best].items() if k not in skip}
                st.json(metrics)

        with tab2:
            st.markdown("#### Perbandingan Semua Model")
            results = final_state.get("model_results", {})
            comparison_data = []
            for model_name, result in results.items():
                skip = {"model_object", "y_test", "y_pred", "X_test"}
                row = {"model": model_name}
                row.update({k: v for k, v in result.items() if k not in skip})
                comparison_data.append(row)
            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

        with tab3:
            st.markdown("#### Semua Keputusan Agent")
            reasoning = final_state.get("reasoning", [])
            for i, r in enumerate(reasoning):
                st.markdown(f"**{i+1}.** {r}")

        with tab4:
            report = final_state.get("report", "")
            if report:
                st.text(report)
            else:
                st.warning("Laporan tidak tersedia")

    # Cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
