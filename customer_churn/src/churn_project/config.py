from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
CV_FOLDS = 5
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
THRESHOLD_OBJECTIVE = "f1"

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_SUMMARY_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost": "XGBoost",
}
