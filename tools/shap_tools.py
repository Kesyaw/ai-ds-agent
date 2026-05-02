import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64


def get_shap_explanation(model, X_test, model_name: str, problem_type: str):
    """
    Generate SHAP values dan return sebagai base64 image + feature importance dict
    """
    try:
        X_sample = X_test.iloc[:min(100, len(X_test))]

        # Pilih explainer berdasarkan model type
        tree_models = ["random_forest", "gradient_boosting", "xgboost", "lightgbm"]

        if any(m in model_name for m in tree_models):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)

        # Handle multiclass (ambil class 1 untuk binary)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values

        # Feature importance dari SHAP
        feature_names = X_sample.columns.tolist()
        importance = np.abs(shap_vals).mean(axis=0)
        importance_dict = dict(sorted(
            zip(feature_names, importance.tolist()),
            key=lambda x: x[1], reverse=True
        ))

        # Plot bar chart
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fdfcff")

        top_n = min(15, len(feature_names))
        top_features = list(importance_dict.keys())[:top_n]
        top_values = list(importance_dict.values())[:top_n]

        # Warna gradient pastel
        colors = []
        for i in range(len(top_features)):
            ratio = i / max(len(top_features) - 1, 1)
            r = int(167 + (251 - 167) * ratio) / 255
            g = int(139 + (146 - 139) * ratio) / 255
            b = int(250 + (114 - 250) * ratio) / 255
            colors.append((r, g, b))

        bars = ax.barh(top_features[::-1], top_values[::-1], color=colors[::-1],
                       height=0.6, edgecolor="none")

        ax.set_xlabel("Mean |SHAP Value|", fontsize=9, color="#8b85a1",
                      fontfamily="sans-serif")
        ax.set_title(f"Feature Importance — {model_name.replace('_',' ').title()}",
                     fontsize=11, fontweight="bold", color="#2d2640",
                     fontfamily="sans-serif", pad=15)

        ax.tick_params(colors="#8b85a1", labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#ede9f8")
        ax.spines["bottom"].set_color("#ede9f8")
        ax.grid(axis="x", color="#f3f0ff", linewidth=1)

        plt.tight_layout()

        # Convert ke base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "image_b64": img_b64,
            "importance_dict": importance_dict,
            "status": "success"
        }

    except Exception as e:
        print(f"  [!] SHAP error: {e}")
        return {
            "image_b64": None,
            "importance_dict": {},
            "status": f"error: {str(e)}"
        }
