# src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "car-recommendation")

# Model parameters
DEFAULT_RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# Feature settings
TARGET_COLUMN = "selling_price"

# Categorical and numerical features
CAT_FEATURES = ["fuel", "seller_type", "transmission", "owner"]
NUM_FEATURES = ["year", "km_driven", "mileage_numeric", "engine_numeric", 
                "max_power_numeric", "seats"]

# Model hyperparameter search space
HYPERPARAMETER_SPACE = {
    "random_forest": {
        "n_estimators": (50, 300),
        "max_depth": (3, 20),
        "min_samples_split": (2, 10),
        "min_samples_leaf": (1, 10)
    },
    "xgboost": {
        "n_estimators": (50, 300),
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 10),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0)
    },
    "lightgbm": {
        "n_estimators": (50, 300),
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 10),
        "num_leaves": (20, 60)
    },
    "catboost": {
        "iterations": (50, 300),
        "learning_rate": (0.01, 0.3),
        "depth": (4, 10)
    }
}