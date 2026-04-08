import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .config import CV_FOLDS, RANDOM_STATE, THRESHOLD_OBJECTIVE, VALIDATION_SIZE
from .models import get_display_name



def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)



def save_plot(path: Path) -> None:
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")



def plot_churn_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    df["Churn"].value_counts().plot(kind="bar", color=["lightblue", "lightcoral"])
    plt.title("Churn Distribution in Dataset")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    save_plot(output_dir / "churn_distribution.png")



def cross_validate_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "average_precision": "average_precision",
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for model_key, pipeline in models.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model_key": model_key, "model": get_display_name(model_key)}
        for metric in scoring:
            row[f"cv_{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"cv_{metric}_std"] = float(scores[f"test_{metric}"].std())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by="cv_roc_auc_mean", ascending=False).reset_index(drop=True)



def calculate_metrics(
    model_key: str,
    y_true: pd.Series,
    y_prob,
    threshold: float = 0.5,
    model_name: str | None = None,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    display_name = model_name or get_display_name(model_key)
    return {
        "model_key": model_key,
        "model": display_name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=True
        ),
    }



def save_evaluation_plots(
    model_key: str,
    model_name: str,
    y_true: pd.Series,
    y_prob,
    y_pred,
    output_dir: Path,
    suffix: str = "",
) -> None:
    suffix_part = f"_{suffix}" if suffix else ""

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"]).plot(cmap="coolwarm")
    plt.title(f"{model_name} Confusion Matrix")
    save_plot(output_dir / f"{model_key}{suffix_part}_confusion_matrix.png")

    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"{model_name} ROC Curve")
    save_plot(output_dir / f"{model_key}{suffix_part}_roc_curve.png")

    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title(f"{model_name} Precision-Recall Curve")
    save_plot(output_dir / f"{model_key}{suffix_part}_pr_curve.png")



def evaluate_model(model_key: str, fitted_pipeline, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> dict:
    display_name = get_display_name(model_key)
    y_prob = fitted_pipeline.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(model_key, y_test, y_prob, threshold=0.5, model_name=display_name)
    y_pred = (y_prob >= 0.5).astype(int)
    save_evaluation_plots(model_key, display_name, y_test, y_prob, y_pred, output_dir)
    return metrics



def find_best_threshold(y_true: pd.Series, y_prob, objective: str = THRESHOLD_OBJECTIVE) -> dict:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return {"threshold": 0.5, "objective": objective, "score": 0.0}

    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if objective == "recall":
        scores = recalls
    else:
        scores = (2 * precisions * recalls) / (precisions + recalls + 1e-12)

    best_index = int(scores.argmax())
    return {
        "threshold": float(thresholds[best_index]),
        "objective": objective,
        "score": float(scores[best_index]),
        "precision": float(precisions[best_index]),
        "recall": float(recalls[best_index]),
    }



def optimize_threshold(model_key: str, pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    X_subtrain, X_valid, y_subtrain, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    tuned_pipeline = pipeline.fit(X_subtrain, y_subtrain)
    y_valid_prob = tuned_pipeline.predict_proba(X_valid)[:, 1]
    threshold_info = find_best_threshold(y_valid, y_valid_prob)
    threshold_info["model_key"] = model_key
    return threshold_info



def evaluate_all_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
):
    leaderboard = cross_validate_models(models, X_train, y_train)
    test_results = []
    fitted_models = {}

    for model_key, pipeline in models.items():
        fitted_pipeline = pipeline.fit(X_train, y_train)
        fitted_models[model_key] = fitted_pipeline
        test_results.append(evaluate_model(model_key, fitted_pipeline, X_test, y_test, output_dir))

    test_results_df = pd.DataFrame(test_results).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
    best_model_key = str(test_results_df.iloc[0]["model_key"])

    threshold_info = optimize_threshold(best_model_key, models[best_model_key], X_train, y_train)
    best_pipeline = fitted_models[best_model_key]
    best_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    tuned_metrics = calculate_metrics(
        best_model_key,
        y_test,
        best_test_prob,
        threshold=threshold_info["threshold"],
        model_name=f"{get_display_name(best_model_key)} (Tuned Threshold)",
    )
    tuned_predictions = (best_test_prob >= threshold_info["threshold"]).astype(int)
    save_evaluation_plots(
        best_model_key,
        tuned_metrics["model"],
        y_test,
        best_test_prob,
        tuned_predictions,
        output_dir,
        suffix="tuned_threshold",
    )

    return leaderboard, test_results_df, best_model_key, fitted_models, tuned_metrics, threshold_info



def save_leaderboard(leaderboard: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "model_comparison.csv"
    leaderboard.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved: {path}")
    return path



def save_results_summary(summary: dict, output_dir: Path) -> Path:
    path = output_dir / "model_results.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")
    return path
