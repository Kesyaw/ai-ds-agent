import io
import json
import pickle
import numpy as np
from datetime import datetime
from agent.state import AgentState


def _safe_convert(obj):
    """Aggressively convert semua tipe ke Python native"""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_safe_convert(i) for i in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_convert(v) for k, v in obj.items()}
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    try:
        json.dumps(obj)
        return obj
    except Exception:
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
    eda_summary = state.get("eda_summary", {}) or {}
    preprocessing_steps = state.get("preprocessing_steps", []) or []
    reasoning = state.get("reasoning", []) or []
    confidence = state.get("confidence_score", 0)

    skip_keys = {"model_object", "y_test", "y_pred", "X_test"}

    metrics = {}
    if best_model_name in model_results:
        for k, v in model_results[best_model_name].items():
            if k not in skip_keys:
                metrics[str(k)] = _safe_convert(v)

    all_models = {}
    for m, r in model_results.items():
        all_models[str(m)] = {}
        for k, v in r.items():
            if k not in skip_keys:
                all_models[str(m)][str(k)] = _safe_convert(v)

    shape = eda_summary.get("shape", {}) or {}

    metadata = {
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": str(best_model_name),
        "confidence_score": _safe_convert(confidence),
        "problem_type": str(state.get("problem_type", "unknown")),
        "target_column": str(state.get("target_column", "unknown")),
        "dataset_info": {
            "rows": _safe_convert(shape.get("rows", 0)),
            "cols": _safe_convert(shape.get("cols", 0)),
            "missing_ratio": _safe_convert(eda_summary.get("overall_missing_ratio", 0)),
            "is_imbalanced": _safe_convert(state.get("is_imbalanced", False)),
        },
        "preprocessing_steps": [str(s) for s in preprocessing_steps],
        "best_model_metrics": metrics,
        "all_models_comparison": all_models,
        "agent_reasoning": [str(r) for r in reasoning],
    }

    # Final safety pass
    metadata = _safe_convert(metadata)

    return json.dumps(metadata, indent=2)
