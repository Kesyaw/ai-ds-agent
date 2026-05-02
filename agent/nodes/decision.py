from agent.state import AgentState


def select_candidate_models(state: AgentState) -> AgentState:
    print("\n[Node] Select Candidate Models dimulai...")

    n_rows = state["n_rows"]
    is_imbalanced = state["is_imbalanced"]
    has_categorical = state["has_categorical"]
    missing_ratio = state["missing_ratio"]
    problem_type = state["problem_type"]

    reasoning = list(state.get("reasoning", []))
    candidates = []

    # === Rule 1: Dataset size ===
    if n_rows < 5000:
        candidates = ["logistic_regression", "random_forest"]
        reasoning.append(
            f"Dataset kecil ({n_rows} rows) -> pilih model ringan: Logistic Regression + Random Forest"
        )
    elif n_rows < 50000:
        candidates = ["logistic_regression", "random_forest", "gradient_boosting"]
        reasoning.append(
            f"Dataset medium ({n_rows} rows) -> coba 3 model: LR + RF + GBM"
        )
    else:
        candidates = ["random_forest", "gradient_boosting", "xgboost"]
        reasoning.append(
            f"Dataset besar ({n_rows} rows) -> fokus ke tree-based ensemble: RF + GBM + XGBoost"
        )

    # === Rule 2: Imbalance ===
    if is_imbalanced:
        if "logistic_regression" not in candidates:
            candidates.insert(0, "logistic_regression")
        reasoning.append(
            "Data imbalanced -> tambahkan Logistic Regression dengan class_weight='balanced'"
        )

    # === Rule 3: Categorical features ===
    if has_categorical:
        if "gradient_boosting" not in candidates:
            candidates.append("gradient_boosting")
        reasoning.append(
            "Ada fitur kategorikal -> Gradient Boosting lebih robust untuk data mixed type"
        )

    # === Rule 4: Selalu ada baseline ===
    if "logistic_regression" not in candidates:
        candidates.insert(0, "logistic_regression")
        reasoning.append(
            "Tambahkan Logistic Regression sebagai baseline - wajib ada untuk perbandingan"
        )

    # === Regression override ===
    if problem_type == "regression":
        candidates = ["linear_regression", "random_forest", "gradient_boosting"]
        reasoning.append(
            "Problem type regression -> ganti kandidat ke: Linear Regression + RF + GBM"
        )

    # Deduplicate, jaga urutan
    seen = set()
    candidates_clean = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            candidates_clean.append(c)

    print(f"  [OK] Kandidat model: {candidates_clean}")
    print(f"  [OK] Total reasoning: {len(reasoning)} poin")

    return {
        **state,
        "candidate_models": candidates_clean,
        "reasoning": reasoning,
    }


def should_optimize(state: AgentState) -> str:
    """
    Edge function untuk LangGraph conditional routing.
    Return: 'optimize' atau 'finalize'
    """
    results = state.get("model_results", {})
    iteration = state.get("iteration_count", 0)
    is_imbalanced = state.get("is_imbalanced", False)
    problem_type = state.get("problem_type", "classification")

    # Max 2 iterasi
    if iteration >= 2:
        print("  [OK] Max iterasi tercapai -> finalize")
        return "finalize"

    if not results:
        return "finalize"

    # Cari score terbaik
    if problem_type == "classification":
        best_score = max(
            results[m].get("f1", 0) for m in results
        )
        threshold = 0.65

        if best_score < threshold and is_imbalanced:
            print(f"  [!] F1 score rendah ({best_score:.2f}) + imbalanced -> optimize")
            return "optimize"

    else:
        # Regression: cek R2 score
        best_score = max(
            results[m].get("r2", 0) for m in results
        )
        threshold = 0.6

        if best_score < threshold:
            print(f"  [!] R2 score rendah ({best_score:.2f}) -> optimize")
            return "optimize"

    print(f"  [OK] Score cukup baik -> finalize")
    return "finalize"
