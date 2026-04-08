import warnings

import pandas as pd

from .config import DATA_PATH, OUTPUT_DIR
from .data import build_numeric_summary, load_data, preprocess, split_data
from .evaluation import (
    ensure_output_dir,
    evaluate_all_models,
    plot_churn_distribution,
    save_leaderboard,
    save_results_summary,
)
from .explainability import compute_feature_importance, generate_shap_outputs, save_feature_importance_outputs
from .insights import build_business_insights, save_business_insights
from .models import build_models, get_display_name

warnings.filterwarnings("ignore")



def build_summary(
    leaderboard: pd.DataFrame,
    test_results: pd.DataFrame,
    best_model_key: str,
    tuned_metrics: dict,
    threshold_info: dict,
    feature_importances: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    insights: dict,
) -> dict:
    return {
        "best_model": {
            "model_key": best_model_key,
            "model": get_display_name(best_model_key),
        },
        "cross_validation": leaderboard.to_dict(orient="records"),
        "test_results": test_results.to_dict(orient="records"),
        "tuned_threshold_result": tuned_metrics,
        "threshold_optimization": threshold_info,
        "feature_importances": feature_importances.head(20).to_dict(),
        "dataset_info": {
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "churn_rate": float(y.mean()),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "business_insights": insights,
    }



def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    df = load_data(DATA_PATH)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nNumeric summary:")
    print(build_numeric_summary(df))

    plot_churn_distribution(df, OUTPUT_DIR)

    X, y = preprocess(df)
    print(f"Feature shape after one-hot encoding: {X.shape}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print("\nOriginal train class distribution:")
    print(y_train.value_counts(normalize=True))

    models = build_models()
    leaderboard, test_results, best_model_key, fitted_models, tuned_metrics, threshold_info = evaluate_all_models(
        models, X_train, y_train, X_test, y_test, OUTPUT_DIR
    )

    print("\nCross-validation leaderboard:")
    print(leaderboard[["model", "cv_roc_auc_mean", "cv_f1_mean", "cv_average_precision_mean"]])
    print("\nTest-set results:")
    print(test_results[["model", "accuracy", "roc_auc", "precision", "recall", "f1", "pr_auc"]])
    print("\nBest-model tuned-threshold result:")
    print(
        pd.DataFrame(
            [
                {
                    "model": tuned_metrics["model"],
                    "threshold": tuned_metrics["threshold"],
                    "accuracy": tuned_metrics["accuracy"],
                    "roc_auc": tuned_metrics["roc_auc"],
                    "precision": tuned_metrics["precision"],
                    "recall": tuned_metrics["recall"],
                    "f1": tuned_metrics["f1"],
                    "pr_auc": tuned_metrics["pr_auc"],
                }
            ]
        )
    )

    best_model = fitted_models[best_model_key]
    feature_importances = compute_feature_importance(best_model, X_test, y_test)
    save_feature_importance_outputs(feature_importances, OUTPUT_DIR)
    generate_shap_outputs(best_model, X_test, OUTPUT_DIR)

    insights = build_business_insights(df, feature_importances)
    save_business_insights(insights, OUTPUT_DIR)

    save_leaderboard(leaderboard, OUTPUT_DIR)
    summary = build_summary(
        leaderboard,
        test_results,
        best_model_key,
        tuned_metrics,
        threshold_info,
        feature_importances,
        X,
        y,
        X_train,
        X_test,
        insights,
    )
    save_results_summary(summary, OUTPUT_DIR)

    print(f"\nBest model: {get_display_name(best_model_key)}")
    print(f"All outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
