import shap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64


def get_shap_explanation(model, X_test, model_name: str, problem_type: str):
    try:
        X_sample = X_test.iloc[:min(30, len(X_test))].copy()
        feature_names = X_sample.columns.tolist()

        # Coba SHAP dulu
        importance_dict = None
        method_used = "shap"

        try:
            tree_models = ["random_forest", "xgboost", "lightgbm"]
            is_simple_tree = any(m in model_name for m in tree_models)

            if is_simple_tree:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            elif "gradient_boosting" in model_name:
                # GradientBoosting pakai KernelExplainer (lebih lambat tapi support multiclass)
                background = shap.maskers.Independent(X_sample, max_samples=10)
                explainer = shap.Explainer(model.predict_proba if hasattr(model, "predict_proba") else model.predict, background)
                shap_values = explainer(X_sample).values
            else:
                explainer = shap.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    sv = np.array(shap_values[1])
                else:
                    sv = np.mean([np.abs(np.array(s)) for s in shap_values], axis=0)
            else:
                sv = np.array(shap_values)

            if sv.ndim == 1:
                sv = sv.reshape(1, -1)
            if sv.ndim == 3:
                sv = sv.mean(axis=2)

            importance = np.abs(sv).mean(axis=0)

            if len(importance) == len(feature_names):
                importance_dict = dict(sorted(
                    zip(feature_names, importance.tolist()),
                    key=lambda x: x[1], reverse=True
                ))

        except Exception:
            importance_dict = None

        # Fallback: permutation importance pakai sklearn
        if importance_dict is None:
            method_used = "permutation"
            try:
                from sklearn.inspection import permutation_importance
                result = permutation_importance(
                    model, X_sample,
                    X_test["__target__"] if "__target__" in X_test.columns else np.zeros(len(X_sample)),
                    n_repeats=3, random_state=42, n_jobs=-1
                )
                importance_dict = dict(sorted(
                    zip(feature_names, result.importances_mean.tolist()),
                    key=lambda x: x[1], reverse=True
                ))
            except Exception:
                importance_dict = None

        # Fallback 2: feature importance dari model langsung
        if importance_dict is None:
            method_used = "model_importance"
            try:
                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_
                    importance_dict = dict(sorted(
                        zip(feature_names, imp.tolist()),
                        key=lambda x: x[1], reverse=True
                    ))
                elif hasattr(model, "coef_"):
                    imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
                    importance_dict = dict(sorted(
                        zip(feature_names, imp.tolist()),
                        key=lambda x: x[1], reverse=True
                    ))
            except Exception:
                pass

        if importance_dict is None:
            return {"image_b64": None, "importance_dict": {}, "status": "error: all methods failed"}

        # Plot
        top_n = min(10, len(importance_dict))
        top_features = list(importance_dict.keys())[:top_n]
        top_values = list(importance_dict.values())[:top_n]

        fig, ax = plt.subplots(figsize=(7, max(3, top_n * 0.35)))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fdfcff")

        colors = []
        for i in range(top_n):
            ratio = i / max(top_n - 1, 1)
            r = (167 + (251 - 167) * ratio) / 255
            g = (139 + (146 - 139) * ratio) / 255
            b = (250 + (114 - 250) * ratio) / 255
            colors.append((r, g, b))

        ax.barh(top_features[::-1], top_values[::-1],
                color=colors[::-1], height=0.55, edgecolor="none")

        label = "SHAP" if method_used == "shap" else "Feature Importance"
        ax.set_xlabel(f"Mean |{label} Value|", fontsize=9, color="#8b85a1")
        ax.set_title(
            f"Feature Importance — {model_name.replace('_', ' ').title()}",
            fontsize=10, fontweight="bold", color="#2d2640", pad=12
        )
        ax.tick_params(colors="#8b85a1", labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#ede9f8")
        ax.spines["bottom"].set_color("#ede9f8")
        ax.grid(axis="x", color="#f3f0ff", linewidth=1)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        plt.clf()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return {"image_b64": img_b64, "importance_dict": importance_dict, "status": "success"}

    except Exception as e:
        plt.close("all")
        return {"image_b64": None, "importance_dict": {}, "status": f"error: {str(e)}"}
