import os
from agent.state import AgentState


def generate_report(state: AgentState) -> str:
    best_model = state.get("best_model", "unknown")
    model_results = state.get("model_results", {})
    preprocessing_steps = state.get("preprocessing_steps", [])
    eda_summary = state.get("eda_summary", {})
    problem_type = state.get("problem_type", "classification")
    is_imbalanced = state.get("is_imbalanced", False)
    confidence_score = state.get("confidence_score", 0.0)
    reasoning = state.get("reasoning", [])

    skip_keys = {"model_object", "y_test", "y_pred", "X_test"}

    best_metrics = {}
    if best_model in model_results:
        best_metrics = {k: v for k, v in model_results[best_model].items() if k not in skip_keys}

    model_comparison = []
    for m, r in model_results.items():
        metrics = {k: round(v, 4) if isinstance(v, float) else v
                   for k, v in r.items() if k not in skip_keys}
        model_comparison.append((m, metrics))

    shape = eda_summary.get("shape", {})
    missing = eda_summary.get("overall_missing_ratio", 0)
    categorical = eda_summary.get("categorical_columns", [])

    metrics_html = ""
    for k, v in best_metrics.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        metrics_html += f'<div class="r-metric"><div class="r-metric-val">{val}</div><div class="r-metric-key">{k.upper()}</div></div>'

    comparison_html = ""
    for m, metrics in model_comparison:
        tag = ' <span class="best-tag">best</span>' if m == best_model else ""
        metric_str = " &nbsp;·&nbsp; ".join(
            f"<b>{k}</b> {round(v,4) if isinstance(v, float) else v}"
            for k, v in metrics.items()
        )
        comparison_html += f'<div class="r-row"><div class="r-model-name">{m.replace("_"," ").title()}{tag}</div><div class="r-metrics-str">{metric_str}</div></div>'

    steps_html = "".join(f'<li>{s}</li>' for s in preprocessing_steps)
    reasoning_html = "".join(
        f'<div class="r-reason"><span class="r-idx">{i+1:02d}</span>{r}</div>'
        for i, r in enumerate(reasoning)
    )

    html = f"""
<style>
.report-wrap {{ font-family: 'Inter', sans-serif; color: #3d3558; }}
.r-section {{ margin-bottom: 1.8rem; }}
.r-section-title {{
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #a78bfa; margin-bottom: 0.7rem;
    padding-bottom: 0.4rem; border-bottom: 1px solid #ede9f8;
}}
.r-summary {{
    background: #f9f7ff; border: 1px solid #ede9f8; border-radius: 10px;
    padding: 1rem 1.2rem; font-size: 0.88rem; line-height: 1.75; color: #5c5578;
}}
.r-metrics-row {{ display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 0.5rem; }}
.r-metric {{
    background: #f3f0ff; border: 1px solid #e0d9f7; border-radius: 10px;
    padding: 0.7rem 1.2rem; text-align: center; min-width: 80px;
}}
.r-metric-val {{ font-size: 1.3rem; font-weight: 700; color: #7c6fcd; }}
.r-metric-key {{ font-size: 0.65rem; color: #a89ec9; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem; }}
.r-list {{ padding-left: 1.2rem; margin: 0; }}
.r-list li {{ font-size: 0.84rem; color: #5c5578; line-height: 1.9; }}
.r-row {{
    padding: 0.7rem 1rem; background: #fdfcff; border: 1px solid #ede9f8;
    border-radius: 8px; margin-bottom: 0.4rem; font-size: 0.82rem;
}}
.r-model-name {{ font-weight: 600; color: #4c4270; margin-bottom: 0.2rem; }}
.r-metrics-str {{ color: #8b85a1; font-size: 0.8rem; }}
.best-tag {{
    background: #a78bfa; color: white; font-size: 0.6rem; font-weight: 600;
    padding: 0.15rem 0.5rem; border-radius: 999px; margin-left: 0.4rem;
    vertical-align: middle; letter-spacing: 0.05em;
}}
.r-reason {{
    display: flex; gap: 0.8rem; align-items: flex-start;
    padding: 0.65rem 0.9rem; background: #fdfcff; border: 1px solid #ede9f8;
    border-left: 3px solid #c4b9f0; border-radius: 0 8px 8px 0;
    margin-bottom: 0.35rem; font-size: 0.82rem; color: #5c5578; line-height: 1.6;
}}
.r-idx {{ color: #a78bfa; font-weight: 700; min-width: 22px; font-size: 0.75rem; padding-top: 0.05rem; }}
.r-confidence {{
    display: inline-block; background: #f3f0ff; border: 1px solid #e0d9f7;
    border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.8rem;
    font-weight: 600; color: #7c6fcd; margin-top: 0.5rem;
}}
</style>

<div class="report-wrap">

<div class="r-section">
<div class="r-section-title">Executive Summary</div>
<div class="r-summary">
Dataset dengan <b>{shape.get('rows','N/A')} baris</b> dan <b>{shape.get('cols','N/A')} kolom</b> berhasil dianalisis.
Problem type: <b>{problem_type}</b>. Dataset {'<b>imbalanced</b>' if is_imbalanced else '<b>balanced</b>'}.
Model terbaik yang dipilih adalah <b>{best_model.replace('_',' ').title()}</b>
dengan confidence score <b>{confidence_score:.0%}</b>.
</div>
</div>

<div class="r-section">
<div class="r-section-title">EDA Findings</div>
<div class="r-summary">
Rows: <b>{shape.get('rows','N/A')}</b> &nbsp;·&nbsp;
Cols: <b>{shape.get('cols','N/A')}</b> &nbsp;·&nbsp;
Missing: <b>{missing:.1%}</b> &nbsp;·&nbsp;
Imbalanced: <b>{is_imbalanced}</b><br>
Categorical columns: <b>{', '.join(categorical) if categorical else 'None'}</b>
</div>
</div>

<div class="r-section">
<div class="r-section-title">Preprocessing Steps</div>
<ul class="r-list">{steps_html}</ul>
</div>

<div class="r-section">
<div class="r-section-title">Model Comparison</div>
{comparison_html}
</div>

<div class="r-section">
<div class="r-section-title">Best Model — {best_model.replace('_',' ').title()}</div>
<div class="r-metrics-row">{metrics_html}</div>
<div class="r-confidence">Confidence {confidence_score:.0%}</div>
</div>

<div class="r-section">
<div class="r-section-title">Agent Reasoning ({len(reasoning)} decisions)</div>
{reasoning_html}
</div>

</div>
"""
    return html


def report_node(state: AgentState) -> AgentState:
    print("\n[Node] Report Generator dimulai...")
    report = generate_report(state)
    print("  [OK] Laporan berhasil dibuat")
    return {**state, "report": report}
