import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from agent.state import AgentState


def preprocessing_node(state: AgentState) -> AgentState:
    print("\n[Node] Preprocessing dimulai...")

    df = state["df_raw"].copy()
    target = state["target_column"]
    problem_type = state["problem_type"]
    is_imbalanced = state["is_imbalanced"]
    missing_ratio = state["missing_ratio"]

    reasoning = list(state.get("reasoning", []))
    preprocessing_steps = []

    # === Pisah fitur dan target ===
    X = df.drop(columns=[target])
    y = df[target]

    # === Step 1: Drop kolom missing > 50% ===
    missing_per_col = X.isnull().sum() / len(X)
    cols_to_drop = missing_per_col[missing_per_col > 0.5].index.tolist()
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        preprocessing_steps.append(f"Drop kolom missing >50%: {cols_to_drop}")
        reasoning.append(f"Drop {len(cols_to_drop)} kolom karena missing >50%: {cols_to_drop}")

    # === Step 2: Pisah kolom numerik dan kategorikal ===
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # === Step 3: Impute missing values ===
    if X[numeric_cols].isnull().any().any():
        strategy = "median" if missing_ratio > 0.1 else "mean"
        imputer = SimpleImputer(strategy=strategy)
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        preprocessing_steps.append(f"Impute numerik dengan {strategy}")
        reasoning.append(f"Impute missing numerik pakai {strategy} (missing ratio: {missing_ratio:.1%})")

    if categorical_cols and X[categorical_cols].isnull().any().any():
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        preprocessing_steps.append("Impute kategorikal dengan most_frequent")

    # === Step 4: Encoding kategorikal ===
    if categorical_cols:
        for col in categorical_cols:
            n_unique = X[col].nunique()
            if n_unique <= 10:
                # One-hot encoding untuk low cardinality
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                preprocessing_steps.append(f"One-hot encoding: {col} ({n_unique} unique)")
            else:
                # Label encoding untuk high cardinality
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                preprocessing_steps.append(f"Label encoding: {col} ({n_unique} unique, high cardinality)")
                reasoning.append(
                    f"Kolom '{col}' punya {n_unique} unique values -> pakai Label Encoding, bukan One-Hot (dimensi meledak)"
                )

    # === Step 5: Scaling numerik ===
    # Update numeric_cols setelah encoding (kolom bisa berubah)
    numeric_cols_final = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols_final:
        scaler = StandardScaler()
        X[numeric_cols_final] = scaler.fit_transform(X[numeric_cols_final])
        preprocessing_steps.append(f"StandardScaler pada {len(numeric_cols_final)} kolom numerik")
        reasoning.append("Apply StandardScaler - penting untuk Logistic Regression dan SVM")

    # === Step 6: Encode target (classification) ===
    if problem_type == "classification":
        if y.dtype == "object":
            le_target = LabelEncoder()
            y = pd.Series(le_target.fit_transform(y), name=target)
            preprocessing_steps.append("Label encode target column")

    # === Gabung kembali ===
    df_processed = X.copy()
    df_processed[target] = y.values

    print(f"  [OK] Shape setelah preprocessing: {df_processed.shape}")
    print(f"  [OK] Steps: {len(preprocessing_steps)}")
    for step in preprocessing_steps:
        print(f"       - {step}")

    return {
        **state,
        "df_processed": df_processed,
        "preprocessing_steps": preprocessing_steps,
        "reasoning": reasoning,
    }
