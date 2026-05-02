import os
from openai import OpenAI
from agent.state import AgentState


def generate_report(state: AgentState) -> str:
    """Generate laporan akhir menggunakan LLM berdasarkan hasil agent"""

    best_model = state.get("best_model", "unknown")
    model_results = state.get("model_results", {})
    reasoning = state.get("reasoning", [])
    preprocessing_steps = state.get("preprocessing_steps", [])
    eda_summary = state.get("eda_summary", {})
    problem_type = state.get("problem_type", "classification")
    is_imbalanced = state.get("is_imbalanced", False)
    confidence_score = state.get("confidence_score", 0.0)
    iteration_count = state.get("iteration_count", 0)

    # === Siapkan metrics best model ===
    best_metrics = {}
    if best_model in model_results:
        result = model_results[best_model]
        skip_keys = {"model_object", "y_test", "y_pred", "X_test"}
        best_metrics = {k: v for k, v in result.items() if k not in skip_keys}

    # === Siapkan perbandingan semua model ===
    model_comparison = []
    for model_name, result in model_results.items():
        skip_keys = {"model_object", "y_test", "y_pred", "X_test"}
        metrics = {k: v for k, v in result.items() if k not in skip_keys}
        model_comparison.append(f"- {model_name}: {metrics}")

    # === Build prompt untuk LLM ===
    prompt = f"""
Kamu adalah seorang ML Engineer senior. Buatkan laporan analisis machine learning yang profesional berdasarkan hasil berikut.

=== INFORMASI DATASET ===
- Jumlah baris: {eda_summary.get('shape', {}).get('rows', 'N/A')}
- Jumlah kolom: {eda_summary.get('shape', {}).get('cols', 'N/A')}
- Problem type: {problem_type}
- Dataset imbalanced: {is_imbalanced}
- Missing ratio: {eda_summary.get('overall_missing_ratio', 0):.1%}
- Kolom kategorikal: {eda_summary.get('categorical_columns', [])}

=== PREPROCESSING ===
{chr(10).join(f'- {step}' for step in preprocessing_steps)}

=== HASIL TRAINING ===
Semua model yang diuji:
{chr(10).join(model_comparison)}

=== BEST MODEL ===
Model terpilih: {best_model}
Metrics: {best_metrics}
Confidence score: {confidence_score:.0%}
Total iterasi: {iteration_count}

=== REASONING AGENT ===
{chr(10).join(f'{i+1}. {r}' for i, r in enumerate(reasoning))}

=== FORMAT LAPORAN ===
Tulis laporan dengan struktur:
1. Ringkasan Eksekutif (2-3 kalimat)
2. Temuan Utama dari EDA
3. Proses Preprocessing yang Dilakukan
4. Perbandingan Model
5. Rekomendasi Model Terbaik dan Alasannya
6. Keterbatasan dan Saran Pengembangan

Tulis dalam Bahasa Indonesia yang profesional dan ringkas.
"""

    # === Panggil LLM ===
    api_key = os.getenv("OPENAI_API_KEY")

    # Kalau tidak ada API key, return template report
    if not api_key or api_key == "your_key_here":
        return _generate_template_report(state, best_model, best_metrics, model_comparison)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3,
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"  [!] LLM error: {e} -> fallback ke template report")
        return _generate_template_report(state, best_model, best_metrics, model_comparison)


def _generate_template_report(state, best_model, best_metrics, model_comparison):
    """Fallback report tanpa LLM"""

    eda = state.get("eda_summary", {})
    shape = eda.get("shape", {})
    confidence = state.get("confidence_score", 0)
    reasoning = state.get("reasoning", [])

    report = f"""
========================================
     LAPORAN ANALISIS ML - AI DS AGENT
========================================

1. RINGKASAN EKSEKUTIF
   Dataset dengan {shape.get('rows', 'N/A')} baris dan {shape.get('cols', 'N/A')} kolom
   berhasil dianalisis. Model terbaik yang dipilih adalah {best_model}
   dengan confidence score {confidence:.0%}.

2. TEMUAN UTAMA EDA
   - Ukuran dataset : {shape.get('rows', 'N/A')} rows x {shape.get('cols', 'N/A')} cols
   - Problem type   : {state.get('problem_type', 'N/A')}
   - Imbalanced     : {state.get('is_imbalanced', False)}
   - Missing ratio  : {eda.get('overall_missing_ratio', 0):.1%}
   - Categorical    : {eda.get('categorical_columns', [])}

3. PREPROCESSING
{chr(10).join(f'   - {s}' for s in state.get('preprocessing_steps', []))}

4. PERBANDINGAN MODEL
{chr(10).join(f'   {m}' for m in model_comparison)}

5. REKOMENDASI
   Model terpilih : {best_model}
   Metrics        : {best_metrics}
   Confidence     : {confidence:.0%}

6. REASONING AGENT
{chr(10).join(f'   {i+1}. {r}' for i, r in enumerate(reasoning))}

========================================
"""
    return report


def report_node(state: AgentState) -> AgentState:
    print("\n[Node] Report Generator dimulai...")

    report = generate_report(state)

    print("  [OK] Laporan berhasil dibuat")
    print("\n" + "="*50)
    print(report)
    print("="*50)

    return {
        **state,
        "report": report,
    }
