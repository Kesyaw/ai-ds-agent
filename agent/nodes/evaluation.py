import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from agent.state import AgentState


def evaluation_node(state: AgentState) -> AgentState:
    print("\n[Node] Evaluation dimulai...")

    model_results = state["model_results"]
    problem_type = state["problem_type"]
    is_imbalanced = state["is_imbalanced"]
    iteration = state.get("iteration_count", 0)

    reasoning = list(state.get("reasoning", []))
    evaluated_results = {}

    for model_name, result in model_results.items():
        print(f"  [..] Evaluasi {model_name}...")

        y_test = result["y_test"]
        y_pred = result["y_pred"]
        model = result["model_object"]

        metrics = {}

        if problem_type == "classification":
            # Tentukan average berdasarkan jumlah kelas
            n_classes = len(np.unique(y_test))
            avg = "binary" if n_classes == 2 else "weighted"

            metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            metrics["f1"] = round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4)
            metrics["precision"] = round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4)
            metrics["recall"] = round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4)

            # ROC-AUC hanya untuk binary
            if n_classes == 2 and hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(result["X_test"])[:, 1]
                metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)

        else:
            # Regression metrics
            metrics["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
            metrics["mae"] = round(mean_absolute_error(y_test, y_pred), 4)
            metrics["r2"] = round(r2_score(y_test, y_pred), 4)

        evaluated_results[model_name] = {
            **result,
            **metrics,
        }

        print(f"  [OK] {model_name}: {metrics}")

    # === Pilih best model ===
    if problem_type == "classification":
        # Kalau imbalanced, prioritaskan F1 bukan accuracy
        primary_metric = "f1" if is_imbalanced else "accuracy"
        reasoning.append(
            f"Primary metric untuk pemilihan model: {primary_metric} "
            f"({'imbalanced dataset' if is_imbalanced else 'balanced dataset'})"
        )
    else:
        primary_metric = "r2"
        reasoning.append("Primary metric untuk regression: R2 score")

    # Pilih model dengan score tertinggi
    if problem_type == "regression":
        # R2 bisa negatif, jadi pakai max normal
        best_model = max(
            evaluated_results,
            key=lambda m: evaluated_results[m].get(primary_metric, -999)
        )
    else:
        best_model = max(
            evaluated_results,
            key=lambda m: evaluated_results[m].get(primary_metric, 0)
        )

    best_score = evaluated_results[best_model].get(primary_metric, 0)

    # === Hitung confidence score ===
    if problem_type == "classification":
        confidence = min(best_score, 1.0)
        if state["n_rows"] < 1000:
            confidence *= 0.8
            reasoning.append(
                "Confidence dikurangi 20% karena dataset terlalu kecil (<1000 rows) - hasil kurang reliable"
            )
    else:
        confidence = max(0.0, min(best_score, 1.0))

    # === Reasoning final ===
    reasoning.append(
        f"Best model: {best_model} dengan {primary_metric}={best_score:.4f}"
    )

    # Bandingkan semua model
    comparison = []
    for m, r in evaluated_results.items():
        score = r.get(primary_metric, 0)
        comparison.append(f"{m}={score:.4f}")
    reasoning.append(f"Perbandingan semua model: {', '.join(comparison)}")

    # === Cek apakah perlu optimasi ===
    needs_optimization = False
    if problem_type == "classification":
        if best_score < 0.65 and is_imbalanced:
            needs_optimization = True
            reasoning.append(
                f"F1 score ({best_score:.4f}) masih rendah di dataset imbalanced -> perlu optimization"
            )
    else:
        if best_score < 0.6:
            needs_optimization = True
            reasoning.append(
                f"R2 score ({best_score:.4f}) masih rendah -> perlu optimization"
            )

    if not needs_optimization:
        reasoning.append(f"Score sudah cukup baik -> tidak perlu optimasi lebih lanjut")

    print(f"\n  [OK] Best model: {best_model} ({primary_metric}={best_score:.4f})")
    print(f"  [OK] Confidence: {confidence:.2f}")
    print(f"  [OK] Needs optimization: {needs_optimization}")

    return {
        **state,
        "model_results": evaluated_results,
        "best_model": best_model,
        "needs_optimization": needs_optimization,
        "confidence_score": confidence,
        "reasoning": reasoning,
    }
