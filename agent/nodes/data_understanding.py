import pandas as pd
import numpy as np
from agent.state import AgentState


def data_understanding_node(state: AgentState) -> AgentState:
    print("\n[Node] Data Understanding dimulai...")

    # === Load dataset ===
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    problem_type = state["problem_type"]

    reasoning = list(state.get("reasoning", []))

    # === Basic info ===
    n_rows, n_cols = df.shape
    n_features = n_cols - 1

    # === Missing values ===
    missing_per_col = df.isnull().sum() / len(df)
    overall_missing = df.isnull().sum().sum() / (n_rows * n_cols)

    # === Categorical check ===
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target in categorical_cols:
        categorical_cols.remove(target)
    has_categorical = len(categorical_cols) > 0

    # === Imbalance check (classification only) ===
    is_imbalanced = False
    class_distribution = {}
    if problem_type == "classification":
        value_counts = df[target].value_counts(normalize=True)
        class_distribution = value_counts.to_dict()
        min_ratio = value_counts.min()
        is_imbalanced = min_ratio < 0.3

    # === Outlier detection (numerik only) ===
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    outlier_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        if outlier_count > 0:
            outlier_info[col] = int(outlier_count)

    # === Bangun EDA Summary ===
    eda_summary = {
        "shape": {"rows": n_rows, "cols": n_cols},
        "target_column": target,
        "problem_type": problem_type,
        "missing_per_column": missing_per_col[missing_per_col > 0].to_dict(),
        "overall_missing_ratio": round(float(overall_missing), 4),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "class_distribution": class_distribution,
        "outlier_info": outlier_info,
    }

    # === Reasoning ===
    reasoning.append(f"Dataset shape: {n_rows} rows, {n_cols} cols")

    if is_imbalanced:
        reasoning.append(
            f"Dataset IMBALANCED - minority class ratio: "
            f"{min(class_distribution.values()):.1%} -> akan pakai class_weight atau SMOTE"
        )
    else:
        if problem_type == "classification":
            reasoning.append("Dataset balanced - tidak perlu resampling")

    if overall_missing > 0.3:
        reasoning.append(
            f"Missing values TINGGI ({overall_missing:.1%}) -> perlu imputation agresif atau drop kolom"
        )
    elif overall_missing > 0:
        reasoning.append(
            f"Missing values rendah ({overall_missing:.1%}) -> cukup simple imputation"
        )
    else:
        reasoning.append("Tidak ada missing values")

    if has_categorical:
        reasoning.append(
            f"Ada {len(categorical_cols)} kolom kategorikal: {categorical_cols} -> perlu encoding"
        )

    if outlier_info:
        reasoning.append(
            f"Ditemukan outlier di kolom: {list(outlier_info.keys())}"
        )

    if n_rows < 5000:
        reasoning.append(
            f"Dataset KECIL ({n_rows} rows) -> hindari model terlalu kompleks"
        )
    elif n_rows < 50000:
        reasoning.append(
            f"Dataset MEDIUM ({n_rows} rows) -> semua model bisa dicoba"
        )
    else:
        reasoning.append(
            f"Dataset BESAR ({n_rows} rows) -> model kompleks seperti XGBoost/LightGBM lebih optimal"
        )

    print(f"  [OK] Shape: {n_rows} rows x {n_cols} cols")
    print(f"  [OK] Missing: {overall_missing:.1%}")
    print(f"  [OK] Imbalanced: {is_imbalanced}")
    print(f"  [OK] Categorical cols: {len(categorical_cols)}")
    print(f"  [OK] Outlier cols: {len(outlier_info)}")
    print(f"  [OK] Reasoning: {len(reasoning)} poin dicatat")

    return {
        **state,
        "df_raw": df,
        "eda_summary": eda_summary,
        "is_imbalanced": is_imbalanced,
        "missing_ratio": float(overall_missing),
        "n_rows": n_rows,
        "n_features": n_features,
        "has_categorical": has_categorical,
        "reasoning": reasoning,
    }
