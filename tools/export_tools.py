import pickle
import io
import json
from datetime import datetime
from agent.state import AgentState


def export_model_pkl(state: AgentState) -> bytes:
    """Export best model sebagai pickle bytes"""
    best_model_name = state.get("best_model")
    model_results = state.get("model_results", {})

    if not best_model_name or best_model_name not in model_results:
        return None

    model_obj = model_results[best_model_name].get("model_object")
    if model_obj is None:
        return None

    buf = io.BytesIO()
    pickle.dump(model_obj, buf)
    buf.seek(0)
    return buf.read()


def export_model_metadata(state: AgentState) -> str:
    """Export metadata model sebagai JSON string"""
    best_model_name = state.get("best_model", "unknown")
    model_results = state.get("model_results", {})
    eda_summary = state.get("eda_summary", {})
    preprocessing_steps = state.get("preprocessing_steps", [])
    reasoning = state.get("reasoning", [])
    confidence = state.get("confidence_score", 0)

    skip_keys = {"model_object", "y_test", "y_pred", "X_test"}
    metrics = {}
    if best_model_name in model_results:
        metrics = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in model_results[best_model_name].items()
            if k not in skip_keys
        }

    all_models = {}
    for m, r in model_results.items():
        all_models[m] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in r.items() if k not in skip_keys
        }

    metadata = {
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": best_model_name,
        "confidence_score": round(confidence, 4),
        "problem_type": state.get("problem_type", "unknown"),
        "target_column": state.get("target_column", "unknown"),
        "dataset_info": {
            "rows": eda_summary.get("shape", {}).get("rows", 0),
            "cols": eda_summary.get("shape", {}).get("cols", 0),
            "missing_ratio": eda_summary.get("overall_missing_ratio", 0),
            "is_imbalanced": state.get("is_imbalanced", False),
        },
        "preprocessing_steps": preprocessing_steps,
        "best_model_metrics": metrics,
        "all_models_comparison": all_models,
        "agent_reasoning": reasoning,
    }

    return json.dumps(metadata, indent=2)
