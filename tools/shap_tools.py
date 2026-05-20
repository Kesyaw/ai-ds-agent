import shap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64


def get_shap_explanation(model, X_test, model_name: str, problem_type: str):
    try:
        # Kurangi sample size drastis untuk hemat memory
        X_sample = X_test.iloc[:min(30, len(X_test))].copy()
        feature_names = X_sample.columns.tolist()

        tree_models = ["random_forest", "gradient_boosting", "xgboost", "lightgbm"]
        is_tree = any(m in model_name for m in tree_models)

        if is_tree:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                sv = np.array(shap_values[1])
            elif len(shap_values) > 2:
                sv = np.mean([np.abs(np.array(s)) for s in shap_values], axis=0)
            else:
                sv = np.array(shap_values[0])
        else:
            sv = np.array(shap_values)

        if sv.ndim == 1:
            sv = sv.reshape(1, -1)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] == 2 else sv.mean(axis=2)

        if len(sv.shape) > 1 and sv.shape[1] != len(feature_names):
            return {"image_b64": None, "importance_dict": {}, "status": "error: shape mismatch"}

        importance = np.abs(sv).mean(axis=0)
        importance_dict = dict(sorted(
            zip(feature_names, importance.tolist()),
            key=lambda x: x[1], reverse=True
        ))

        # Plot kecil untuk hemat memory
        top_n = min(10, len(feature_names))
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
        ax.set_xlabel("Mean |SHAP Value|", fontsize=9, color="#8b85a1")
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
        # Turunkan DPI untuk hemat memory
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
