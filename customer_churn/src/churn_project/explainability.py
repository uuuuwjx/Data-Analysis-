import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from .evaluation import save_plot



def compute_feature_importance(fitted_pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
    model = fitted_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
    elif hasattr(model, "coef_"):
        importances = pd.Series(np.abs(model.coef_[0]), index=X_test.columns)
    else:
        result = permutation_importance(
            fitted_pipeline,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=1,
            scoring="roc_auc",
        )
        importances = pd.Series(result.importances_mean, index=X_test.columns)

    return importances.sort_values(ascending=False)



def save_feature_importance_outputs(feature_importances: pd.Series, output_dir):
    feature_importance_df = feature_importances.reset_index()
    feature_importance_df.columns = ["feature", "importance"]
    csv_path = output_dir / "feature_importances.csv"
    feature_importance_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved: {csv_path}")

    top_features = feature_importances.head(15).sort_values()
    plt.figure(figsize=(10, 6))
    top_features.plot(kind="barh")
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    save_plot(output_dir / "feature_importances.png")



def generate_shap_outputs(fitted_pipeline, X_test: pd.DataFrame, output_dir):
    model = fitted_pipeline.named_steps["model"]
    try:
        import shap

        X_numeric = X_test.astype(float)

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_input = X_numeric.to_numpy()
            shap_values = explainer.shap_values(shap_input)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif getattr(shap_values, "ndim", 0) == 3:
                shap_values = shap_values[:, :, 1]

            shap.summary_plot(shap_values, shap_input, feature_names=X_numeric.columns, show=False)
            plt.title("SHAP Summary")
            plt.tight_layout()
            save_plot(output_dir / "shap_summary.png")

            shap.summary_plot(
                shap_values,
                shap_input,
                feature_names=X_numeric.columns,
                plot_type="bar",
                show=False,
            )
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            save_plot(output_dir / "shap_feature_importance.png")
        else:
            print("SHAP summary is only generated for tree-based best models.")
    except Exception as exc:
        print(f"SHAP analysis skipped: {exc}")
