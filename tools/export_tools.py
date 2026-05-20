import io
import json
import pickle
import numpy as np
from datetime import datetime
from agent.state import AgentState


def _convert(obj):
    """Convert semua tipe non-serializable ke Python native"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


def export_model_pkl(state: AgentState) -> bytes:
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
    best_model_name = state.get("best_model", "unknown")
    model_results = state.get("model_results", {})
    eda_summary = state.get("eda_summary", {})
    preprocessing_steps = state.get("preprocessing_steps", [])
    reasoning = state.get("reasoning", [])
    confidence = state.get("confidence_score", 0)

    skip_keys = {"model_object", "y_test", "y_pred", "X_test"}

    metrics = {}
    if best_model_name in model_results:
        for k, v in model_results[best_model_name].items():
            if k not in skip_keys:
                metrics[k] = round(float(v), 4) if isinstance(v, (float, np.floating)) else _convert(v)

    all_models = {}
    for m, r in model_results.items():
        all_models[m] = {}
        for k, v in r.items():
            if k not in skip_keys:
                all_models[m][k] = round(float(v), 4) if isinstance(v, (float, np.floating)) else _convert(v)

    shape = eda_summary.get("shape", {})

    metadata = {
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": str(best_model_name),
        "confidence_score": round(float(confidence), 4),
        "problem_type": str(state.get("problem_type", "unknown")),
        "target_column": str(state.get("target_column", "unknown")),
        "dataset_info": {
            "rows": int(shape.get("rows", 0)),
            "cols": int(shape.get("cols", 0)),
            "missing_ratio": float(eda_summary.get("overall_missing_ratio", 0)),
            "is_imbalanced": bool(state.get("is_imbalanced", False)),
        },
        "preprocessing_steps": [str(s) for s in preprocessing_steps],
        "best_model_metrics": metrics,
        "all_models_comparison": all_models,
        "agent_reasoning": [str(r) for r in reasoning],
    }

    return json.dumps(metadata, indent=2, default=_convert)
