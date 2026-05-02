from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict):
    # === INPUT ===
    dataset_path: str
    target_column: str
    problem_type: str              # "classification" atau "regression"

    # === DATA ===
    df_raw: Optional[Any]          # DataFrame original
    df_processed: Optional[Any]    # DataFrame setelah preprocessing

    # === EDA RESULTS ===
    eda_summary: Optional[Dict]
    is_imbalanced: bool
    missing_ratio: float
    n_rows: int
    n_features: int
    has_categorical: bool

    # === PREPROCESSING ===
    preprocessing_steps: List[str]  # log langkah yang diambil

    # === MODELING ===
    candidate_models: List[str]
    model_results: Dict[str, Dict]  # {model_name: {metric: value, model_object: ...}}

    # === DECISION ===
    best_model: Optional[str]
    needs_optimization: bool
    iteration_count: int

    # === REASONING LOG ===
    reasoning: List[str]           # semua keputusan agent dicatat di sini

    # === OUTPUT ===
    report: Optional[str]
    confidence_score: float        # 0.0 - 1.0