# main.py

import pandas as pd
import numpy as np
import mlflow
import argparse
import logging
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, TARGET_COLUMN
from src.data.data_processing import load_data, preprocess_data, split_data, save_processed_data
from src.features.feature_engineering import prepare_features_targets
from src.models.train_model import compare_models, find_best_model, register_best_model,find_best_run_and_register

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("car_recommendation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Car Recommendation System")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=None,
        help="Path to the car data CSV file"
    )
    parser.add_argument(
        "--optimize", 
        action="store_true", 
        help="Optimize hyperparameters for each model"
    )
    parser.add_argument(
        "--trials", 
        type=int, 
        default=30,
        help="Number of hyperparameter optimization trials"
    )
    parser.add_argument(
        "--register_model", 
        action="store_true", 
        help="Register the best model to MLflow Model Registry"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the car recommendation pipeline."""
    # Parse arguments
    args = parse_args()
    
    logger.info("Starting Car Recommendation System pipeline")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Step 1: Load and preprocess data
    logger.info("Step 1: Loading and preprocessing data")
    df = load_data(args.data_path)
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    preprocessed_df = preprocess_data(df)
    
    # Step 2: Split data
    logger.info("Step 2: Splitting data into train, validation, and test sets")
    train_df, val_df, test_df = split_data(preprocessed_df)
    save_processed_data(train_df, val_df, test_df)
    
    # Step 3: Prepare features and targets
    logger.info("Step 3: Preparing features and targets")
    X_train, y_train = prepare_features_targets(train_df, TARGET_COLUMN)
    X_val, y_val = prepare_features_targets(val_df, TARGET_COLUMN)
    X_test, y_test = prepare_features_targets(test_df, TARGET_COLUMN)
    
    # Step 4: Train and compare models 
    logger.info("Step 4: Training and comparing models")
    results = compare_models(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        optimize=args.optimize, n_trials=args.trials
    )


    # Step 5: Find best model
    
    try:
        # Check connection to MLflow server
        client = MlflowClient()
        experiments = mlflow.search_experiments()
        print(f"Successfully connected to MLflow server with {len(experiments)} experiments")
    except Exception as e:
        print(f"Error connecting to MLflow server: {e}")
        print("Please ensure your MLflow server is running at http://localhost:5000")
        sys.exit(1)
    
   ## Example 1: Find best model across specific experiments
   #model_name, version, best_run = find_best_run_and_register(
   #    experiment_names=["experiment1", "experiment2"],  # Replace with your experiment names
   #    metric_name="accuracy",  # Replace with your metric name
   #    higher_is_better=True,
   #    model_name="best_classification_model",  # Replace with your desired model name
   #    tags={"source": "auto_selection", "algorithm_type": "classification"},
   #    description="Best model automatically selected based on accuracy"
   #)
    
    # Example 2: Find best model across ALL experiments
    model_name, version, best_run = find_best_run_and_register(
        metric_name="test_rmse",
        higher_is_better=True,
        model_name="best_overall_model",
        description="Best model across all experiments"
    )
    
    """
    # Step 5: Find best model
    logger.info("Step 5: Finding the best model")
    best_model_name, best_model_results = find_best_model(results)
    
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Validation RMSE: {best_model_results['val_metrics']['rmse']:.2f}")
    logger.info(f"Test RMSE: {best_model_results['test_metrics']['rmse']:.2f}")
    logger.info(f"Test R²: {best_model_results['test_metrics']['r2']:.4f}")
    
    # Step 6: Register best model (if requested)
    if args.register_model:
        logger.info("Step 6: Registering the best model to MLflow Model Registry")
        register_best_model(best_model_results['model'],best_model_name)
    
    logger.info("Car Recommendation System pipeline completed successfully")
"""

if __name__ == "__main__":
    main()
