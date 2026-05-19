from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import build_graph

app = FastAPI(
    title="AI Data Science Agent API",
    description="REST API for the AI DS Agent — upload dataset, train, and predict",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory model store (per session)
model_store = {}


# === SCHEMAS ===
class TrainRequest(BaseModel):
    target_column: str
    problem_type: str = "classification"


class PredictRequest(BaseModel):
    session_id: str
    data: list  # list of dicts, each dict = 1 row


# === ENDPOINTS ===

@app.get("/")
def root():
    return {
        "name": "AI Data Science Agent API",
        "version": "1.0.0",
        "endpoints": [
            "POST /train  — upload CSV + config, run agent",
            "POST /predict — send data, get prediction",
            "GET  /status/{session_id} — get training results",
            "GET  /health — health check",
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
async def train(
    file: UploadFile = File(...),
    target_column: str = "target",
    problem_type: str = "classification"
):
    """
    Upload CSV dataset dan jalankan agent pipeline.
    Returns session_id untuk predict endpoint.
    """
    # Validasi file
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Simpan file temp
    content = await file.read()
    temp_path = f"temp_api_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(content)

    # Validasi kolom
    try:
        df_check = pd.read_csv(temp_path)
        if target_column not in df_check.columns:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail=f"Column '{target_column}' not found. Available: {df_check.columns.tolist()}"
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Jalankan agent
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

    try:
        graph = build_graph()
        final_state = None
        for step in graph.stream(initial_state):
            node_name = list(step.keys())[0]
            final_state = step[node_name]
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Simpan model ke store
    import uuid
    session_id = str(uuid.uuid4())[:8]

    best_model_name = final_state.get("best_model")
    model_results = final_state.get("model_results", {})
    model_obj = None
    if best_model_name and best_model_name in model_results:
        model_obj = model_results[best_model_name].get("model_object")

    skip_keys = {"model_object", "y_test", "y_pred", "X_test"}
    metrics = {}
    if best_model_name and best_model_name in model_results:
        metrics = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in model_results[best_model_name].items()
            if k not in skip_keys
        }

    # Ambil feature columns SETELAH preprocessing (dari X_test)
    processed_feature_cols = []
    if best_model_name and best_model_name in model_results:
        x_test_obj = model_results[best_model_name].get("X_test")
        if x_test_obj is not None:
            processed_feature_cols = x_test_obj.columns.tolist()

    model_store[session_id] = {
        "model": model_obj,
        "best_model_name": best_model_name,
        "target_column": target_column,
        "problem_type": problem_type,
        "feature_columns": processed_feature_cols,
        "raw_feature_columns": [c for c in df_check.columns if c != target_column],
        "preprocessing_steps": final_state.get("preprocessing_steps", []),
        "metrics": metrics,
        "confidence_score": final_state.get("confidence_score", 0),
        "reasoning": final_state.get("reasoning", []),
        "eda_summary": final_state.get("eda_summary", {}),
    }

    return {
        "session_id": session_id,
        "best_model": best_model_name,
        "confidence_score": round(final_state.get("confidence_score", 0), 4),
        "metrics": metrics,
        "preprocessing_steps": final_state.get("preprocessing_steps", []),
        "message": f"Training complete. Use session_id '{session_id}' to predict."
    }


@app.get("/status/{session_id}")
def get_status(session_id: str):
    """Get training results by session_id"""
    if session_id not in model_store:
        raise HTTPException(status_code=404, detail="Session not found")

    store = model_store[session_id]
    return {
        "session_id": session_id,
        "best_model": store["best_model_name"],
        "confidence_score": store["confidence_score"],
        "metrics": store["metrics"],
        "preprocessing_steps": store["preprocessing_steps"],
        "feature_columns": store["feature_columns"],
        "reasoning": store["reasoning"],
        "eda_summary": store["eda_summary"],
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Predict menggunakan model yang sudah ditraining.
    Body: { session_id, data: [{col1: val1, col2: val2, ...}] }
    """
    session_id = request.session_id

    if session_id not in model_store:
        raise HTTPException(status_code=404, detail="Session not found. Train first via POST /train")

    store = model_store[session_id]
    model = store["model"]

    if model is None:
        raise HTTPException(status_code=500, detail="Model not available in this session")

    try:
        df_input = pd.DataFrame(request.data)
        feature_cols = store["feature_columns"]

        # Reindex ke kolom yang dipakai saat training
        # Kolom yang tidak ada diisi 0
        for col in feature_cols:
            if col not in df_input.columns:
                df_input[col] = 0

        # Hanya ambil kolom yang dipakai saat training, urutan sama
        df_input = df_input[feature_cols]

        # Handle missing
        df_input = df_input.fillna(0)

        predictions = model.predict(df_input).tolist()

        probabilities = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_input)
            probabilities = proba.tolist()

        return {
            "session_id": session_id,
            "model_used": store["best_model_name"],
            "predictions": predictions,
            "probabilities": probabilities,
            "n_samples": len(predictions),
            "note": "Send preprocessed features or use feature_columns from /status endpoint"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Hapus session dari memory"""
    if session_id not in model_store:
        raise HTTPException(status_code=404, detail="Session not found")
    del model_store[session_id]
    return {"message": f"Session {session_id} deleted"}
