import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from agent.state import AgentState


PARAM_GRIDS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"],
        "max_iter": [500, 1000],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "linear_regression": {},
}


def get_base_model(model_name: str, problem_type: str, is_imbalanced: bool):
    class_weight = "balanced" if is_imbalanced else None

    if problem_type == "classification":
        models = {
            "logistic_regression": LogisticRegression(
                class_weight=class_weight, random_state=42
            ),
            "random_forest": RandomForestClassifier(
                class_weight=class_weight, random_state=42
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "xgboost": XGBClassifier(
                random_state=42, verbosity=0,
                scale_pos_weight=10 if is_imbalanced else 1
            ),
        }
    else:
        models = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(random_state=42),
            "gradient_boosting": GradientBoostingRegressor(random_state=42),
            "xgboost": XGBRegressor(random_state=42, verbosity=0),
        }

    return models.get(model_name)


def optimization_node(state: AgentState) -> AgentState:
    print("\n[Node] Optimization dimulai...")

    df = state["df_processed"]
    target = state["target_column"]
    problem_type = state["problem_type"]
    is_imbalanced = state["is_imbalanced"]
    best_model_name = state["best_model"]
    model_results = state["model_results"]
    iteration = state.get("iteration_count", 0)

    reasoning = list(state.get("reasoning", []))

    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if problem_type == "classification" else None
    )

    # === Scoring metric ===
    scoring = "f1_weighted" if problem_type == "classification" else "r2"
    if is_imbalanced:
        scoring = "f1_weighted"

    # === Tuning best model ===
    param_grid = PARAM_GRIDS.get(best_model_name, {})

    if not param_grid:
        print(f"  [!] Tidak ada param grid untuk {best_model_name} -> skip tuning")
        reasoning.append(f"Skip tuning {best_model_name} - tidak ada hyperparameter yang bisa dioptimasi")
        return {**state, "reasoning": reasoning}

    base_model = get_base_model(best_model_name, problem_type, is_imbalanced)

    print(f"  [..] RandomizedSearchCV untuk {best_model_name}...")
    print(f"       Scoring: {scoring}, n_iter=10, cv=3")

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=10,
        scoring=scoring,
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_estimator = search.best_estimator_

    # === Evaluasi sebelum vs sesudah ===
    y_pred_tuned = best_estimator.predict(X_test)

    if problem_type == "classification":
        from sklearn.metrics import f1_score
        avg = "binary" if len(set(y_test)) == 2 else "weighted"
        score_before = model_results[best_model_name].get("f1", 0)
        score_after = round(f1_score(y_test, y_pred_tuned, average=avg, zero_division=0), 4)
        metric_name = "f1"
    else:
        from sklearn.metrics import r2_score
        score_before = model_results[best_model_name].get("r2", 0)
        score_after = round(r2_score(y_test, y_pred_tuned), 4)
        metric_name = "r2"

    improvement = score_after - score_before
    
    # === Update model results dengan versi tuned ===
    model_results[best_model_name].update({
        "model_object": best_estimator,
        "y_pred": y_pred_tuned,
        metric_name: score_after,
        "best_params": best_params,
        "tuned": True,
    })

    # === Reasoning ===
    reasoning.append(
        f"Tuning {best_model_name}: {metric_name} {score_before:.4f} -> {score_after:.4f} "
        f"(improvement: {improvement:+.4f})"
    )
    reasoning.append(f"Best params setelah tuning: {best_params}")

    if improvement > 0:
        reasoning.append(f"Tuning berhasil meningkatkan performa sebesar {improvement:.4f}")
    else:
        reasoning.append(f"Tuning tidak meningkatkan performa - model default sudah optimal")

    print(f"  [OK] {metric_name} sebelum: {score_before:.4f}")
    print(f"  [OK] {metric_name} sesudah: {score_after:.4f}")
    print(f"  [OK] Improvement: {improvement:+.4f}")
    print(f"  [OK] Best params: {best_params}")

    return {
        **state,
        "model_results": model_results,
        "reasoning": reasoning,
    }
