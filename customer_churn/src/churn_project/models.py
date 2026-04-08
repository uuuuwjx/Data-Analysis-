from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from .config import MODEL_DISPLAY_NAMES, RANDOM_STATE

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency at runtime
    XGBClassifier = None


def build_smote():
    return SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=NearestNeighbors(n_neighbors=5, algorithm="ball_tree", n_jobs=1),
    )


def build_models():
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("smote", build_smote()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("smote", build_smote()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_split=6,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("smote", build_smote()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_depth=6,
                        max_iter=250,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = Pipeline(
            steps=[
                ("smote", build_smote()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=4,
                        min_child_weight=3,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    return models

def get_display_name(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)
