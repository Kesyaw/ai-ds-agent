import json
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from agent.state import AgentState


def log_experiment(run_name: str, params: dict, metrics: dict = {}):
    """Simple logging ke JSON, pengganti MLflow untuk cloud"""
    log_dir = "tracking"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "experiments.json")

    # Convert numpy types ke Python native
    def convert(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return str(obj)

    entry = {"run_name": run_name, "params": params, "metrics": metrics}

    existing = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                existing = json.load(f)
        except:
            existing = []

    existing.append(entry)
    with open(log_file, "w") as f:
        json.dump(existing, f, indent=2, default=convert)
        

def get_model(model_name: str, problem_type: str, is_imbalanced: bool):
    if problem_type == "classification":
        class_weight = "balanced" if is_imbalanced else None
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight=class_weight, random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100, class_weight=class_weight, random_state=42
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            "xgboost": XGBClassifier(
                n_estimators=100, random_state=42, verbosity=0,
                scale_pos_weight=10 if is_imbalanced else 1
            ),
        }
    else:
        models = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "xgboost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        }
    return models.get(model_name)


def modeling_node(state: AgentState) -> AgentState:
    print("\n[Node] Modeling dimulai...")

    df = state["df_processed"]
    target = state["target_column"]
    problem_type = state["problem_type"]
    is_imbalanced = state["is_imbalanced"]
    candidate_models = state["candidate_models"]
    iteration = state.get("iteration_count", 0)
    reasoning = list(state.get("reasoning", []))

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if problem_type == "classification" and y.nunique() < len(y) * 0.5 else None
    )

    print(f"  [OK] Train: {X_train.shape}, Test: {X_test.shape}")

    model_results = {}

    for model_name in candidate_models:
        print(f"  [..] Training {model_name}...")

        model = get_model(model_name, problem_type, is_imbalanced)
        if model is None:
            print(f"  [!!] Model {model_name} tidak ditemukan, skip")
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        log_experiment(
            run_name=f"{model_name}_iter{iteration}",
            params={"model": model_name, "iteration": iteration,
                    "n_train": len(X_train), "is_imbalanced": is_imbalanced}
        )

        model_results[model_name] = {
            "model_object": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "X_test": X_test,
        }

        print(f"  [OK] {model_name} selesai")

    reasoning.append(
        f"Iterasi {iteration}: training {len(model_results)} model -> {list(model_results.keys())}"
    )

    return {
        **state,
        "model_results": model_results,
        "iteration_count": iteration + 1,
        "reasoning": reasoning,
    }
