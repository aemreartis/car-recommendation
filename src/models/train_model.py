# src/models/train_model.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging
import joblib
import time
import optuna
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from ..config import (
    MLFLOW_TRACKING_URI, 
    MLFLOW_EXPERIMENT_NAME, 
    MODELS_DIR, 
    DEFAULT_RANDOM_STATE,
    HYPERPARAMETER_SPACE
)
from ..features.feature_engineering import create_preprocessing_pipeline

# Configure logging
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def evaluate_model(model, X, y):
    """
    Evaluate a model's performance.
    
    Parameters:
    -----------
    model : estimator
        Trained model
    X : array-like
        Input features
    y : array-like
        Target values
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics

def create_pipeline(model, preprocessor=None):
    """
    Create a full model pipeline.
    
    Parameters:
    -----------
    model : estimator
        Machine learning model
    preprocessor : transformer, optional
        Data preprocessor
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Full pipeline
    """
    if preprocessor is None:
        preprocessor = create_preprocessing_pipeline()
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

def objective(trial, X_train, y_train, X_val, y_val, model_type):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Parameters:
    -----------
    trial : optuna.trial.Trial
        Optuna trial object
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation data
    model_type : str
        Type of model to optimize
        
    Returns:
    --------
    float
        Validation RMSE
    """
    # Create preprocessor
    preprocessor = create_preprocessing_pipeline()
    
    # Get hyperparameter space for this model type
    space = HYPERPARAMETER_SPACE[model_type]
    
    # Define parameters based on model type
    if model_type == 'random_forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *space['n_estimators']),
            'max_depth': trial.suggest_int('max_depth', *space['max_depth']),
            'min_samples_split': trial.suggest_int('min_samples_split', *space['min_samples_split']),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', *space['min_samples_leaf']),
            'random_state': DEFAULT_RANDOM_STATE
        }
        model = RandomForestRegressor(**params)
    
    elif model_type == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *space['n_estimators']),
            'learning_rate': trial.suggest_float('learning_rate', *space['learning_rate']),
            'max_depth': trial.suggest_int('max_depth', *space['max_depth']),
            'subsample': trial.suggest_float('subsample', *space['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *space['colsample_bytree']),
            'random_state': DEFAULT_RANDOM_STATE
        }
        model = xgb.XGBRegressor(**params)
    
    elif model_type == 'lightgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *space['n_estimators']),
            'learning_rate': trial.suggest_float('learning_rate', *space['learning_rate']),
            'max_depth': trial.suggest_int('max_depth', *space['max_depth']),
            'num_leaves': trial.suggest_int('num_leaves', *space['num_leaves']),
            'random_state': DEFAULT_RANDOM_STATE
        }
        model = lgb.LGBMRegressor(**params)
    
    elif model_type == 'catboost':
        params = {
            'iterations': trial.suggest_int('iterations', *space['iterations']),
            'learning_rate': trial.suggest_float('learning_rate', *space['learning_rate']),
            'depth': trial.suggest_int('depth', *space['depth']),
            'random_state': DEFAULT_RANDOM_STATE
        }
        model = CatBoostRegressor(**params, verbose=0)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create pipeline
    pipeline = create_pipeline(model, preprocessor)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    metrics = evaluate_model(pipeline, X_val, y_val)
    
    return metrics['rmse']

def optimize_hyperparameters(X_train, y_train, X_val, y_val, model_type, n_trials=30):
    """
    Optimize hyperparameters using Optuna.
    
    Parameters:
    -----------
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation data
    model_type : str
        Type of model to optimize
    n_trials : int, optional
        Number of optimization trials
    
    Returns:
    --------
    dict
        Optimized parameters
    """
    logger.info(f"Optimizing hyperparameters for {model_type} with {n_trials} trials")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, model_type),
        n_trials=n_trials
    )
    
    logger.info(f"Best hyperparameters for {model_type}: {study.best_params}")
    logger.info(f"Best validation RMSE: {study.best_value:.2f}")
    
    return study.best_params

def train_model(X_train, y_train, model_type='random_forest', params=None):
    """
    Train a model with specified hyperparameters.
    
    Parameters:
    -----------
    X_train, y_train : array-like
        Training data
    model_type : str
        Type of model to train
    params : dict, optional
        Model hyperparameters
    
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Trained model pipeline
    """
    if params is None:
        params = {}
    
    # Set default random state
    params['random_state'] = DEFAULT_RANDOM_STATE
    
    # Create model based on type
    if model_type == 'random_forest':
        model = RandomForestRegressor(**params)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(**params)
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(**params)
    elif model_type == 'catboost':
        model = CatBoostRegressor(**params, verbose=0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create pipeline
    pipeline = create_pipeline(model)
    
    # Train model
    logger.info(f"Training {model_type} model")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    return pipeline

def run_training_experiment(model_type, X_train, y_train, X_val, y_val, X_test, y_test, optimize=False, n_trials=30):
    """
    Run a complete training experiment with MLflow tracking.
    
    Parameters:
    -----------
    model_type : str
        Type of model to train
    X_train, y_train, X_val, y_val, X_test, y_test : array-like
        Training, validation, and test data
    optimize : bool, optional
        Whether to optimize hyperparameters
    n_trials : int, optional
        Number of optimization trials
    
    Returns:
    --------
    dict
        Dictionary with results of the experiment
    """
    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_type}_experiment"):
        # Optimize hyperparameters if requested
        if optimize:
            params = optimize_hyperparameters(
                X_train, y_train, X_val, y_val, model_type, n_trials
            )
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
        else:
            params = {'random_state': DEFAULT_RANDOM_STATE}
        
        # Train model
        pipeline = train_model(X_train, y_train, model_type, params)
        
        # Evaluate on validation data
        val_metrics = evaluate_model(pipeline, X_val, y_val)
        
        # Log validation metrics
        for metric_name, metric_value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)
            
        # Evaluate on test data
        test_metrics = evaluate_model(pipeline, X_test, y_test)
        
        # Log test metrics
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        # Log the model
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Save model locally
        model_path = MODELS_DIR / f"{model_type}_model.joblib"
        joblib.dump(pipeline, model_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Validation RMSE: {val_metrics['rmse']:.2f}")
        logger.info(f"Test RMSE: {test_metrics['rmse']:.2f}")
        
        results = {
            'model': pipeline,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model_path': model_path
        }
        
        return results

def compare_models(X_train, y_train, X_val, y_val, X_test, y_test, optimize=True, n_trials=30):
    """
    Train and compare multiple model types.
    
    Parameters:
    -----------
    X_train, y_train, X_val, y_val, X_test, y_test : array-like
        Training, validation, and test data
    optimize : bool, optional
        Whether to optimize hyperparameters
    n_trials : int, optional
        Number of optimization trials
    
    Returns:
    --------
    dict
        Dictionary with results for all models
    """
    model_types = ['random_forest', 'xgboost', 'lightgbm', 'catboost']
    results = {}
    
    for model_type in model_types:
        logger.info(f"\n{'='*50}\nTraining {model_type}\n{'='*50}")
        
        model_results = run_training_experiment(
            model_type, X_train, y_train, X_val, y_val, X_test, y_test, 
            optimize=optimize, n_trials=n_trials
        )
        
        results[model_type] = model_results
    
    return results

def find_best_model(results):
    """
    Find the best model based on validation RMSE.
    
    Parameters:
    -----------
    results : dict
        Dictionary with results for all models
    
    Returns:
    --------
    tuple
        (best_model_name, best_model_results)
    """
    best_model_name = None
    best_rmse = float('inf')
    
    for model_name, model_results in results.items():
        val_rmse = model_results['val_metrics']['rmse']
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model_name = model_name
    
    return best_model_name, results[best_model_name]

def register_best_model(model, model_name="car-recommendation-model"):
    """
    Register the best model to MLflow Model Registry.
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        Best model pipeline
    model_name : str, optional
        Name for the registered model
    
    Returns:
    --------
    mlflow.entities.model_registry.ModelVersion
        Registered model version
    """
    with mlflow.start_run():
        model_uri = mlflow.sklearn.log_model(model, "model")
        model_version = mlflow.register_model(model_uri, model_name)
        
        logger.info(f"Model registered as '{model_name}' with version {model_version.version}")
        
        return model_version


def set_mlflow_tracking_uri():
    """Set the MLflow tracking URI to the local server"""
    mlflow.set_tracking_uri("http://localhost:5000")
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

def find_best_run_and_register(
    experiment_names=None,
    experiment_ids=None,
    metric_name="accuracy",
    higher_is_better=True,
    model_name="best_model",
    tags=None,
    description=None
):
    """
    Find the best run across specified experiments based on a metric and register it.
    
    Parameters:
    -----------
    experiment_names : list, optional
        Names of experiments to search through
    experiment_ids : list, optional
        IDs of experiments to search through
    metric_name : str, default="accuracy"
        The metric to use for comparison
    higher_is_better : bool, default=True
        Whether higher values of the metric are better
    model_name : str, default="best_model"
        Name to register the model under
    tags : dict, optional
        Tags to add to the registered model version
    description : str, optional
        Description for the registered model version
        
    Returns:
    --------
    tuple
        (registered model name, version, best run info)
    """
    client = MlflowClient()
    
    # Get experiment IDs from names if provided
    if experiment_names and not experiment_ids:
        experiment_ids = []
        for name in experiment_names:
            exp = mlflow.get_experiment_by_name(name)
            if exp:
                experiment_ids.append(exp.experiment_id)
    
    # If neither experiment names nor IDs are provided, get all experiments
    if not experiment_ids and not experiment_names:
        experiments = mlflow.search_experiments()
        experiment_ids = [exp.experiment_id for exp in experiments]
        print(f"Searching across all {len(experiment_ids)} experiments")
    
    if not experiment_ids:
        raise ValueError("No valid experiments found. Please check your MLflow tracking server connection.")
    
    # Collect all runs from the specified experiments
    all_runs = []
    for exp_id in experiment_ids:
        runs = mlflow.search_runs(experiment_ids=[exp_id])
        all_runs.append(runs)
    
    if not all_runs:
        raise ValueError("No runs found in the specified experiments.")
    
    # Combine all runs into a single DataFrame
    runs_df = pd.concat(all_runs, ignore_index=True)
    
    # Filter out runs that don't have the metric
    metric_col = f"metrics.{metric_name}"
    runs_df = runs_df[runs_df[metric_col].notnull()]
    
    if runs_df.empty:
        raise ValueError(f"No runs found with the metric '{metric_name}'.")
    
    # Find the best run based on the metric
    if higher_is_better:
        best_run_idx = runs_df[metric_col].idxmax()
    else:
        best_run_idx = runs_df[metric_col].idxmin()
    
    best_run = runs_df.iloc[best_run_idx]
    best_run_id = best_run["run_id"]
    best_metric_value = best_run[metric_col]
    
    print(f"Best run found: {best_run_id} with {metric_name} = {best_metric_value}")
    print(f"Experiment ID: {best_run['experiment_id']}")
    
    # Get the run object
    run = mlflow.get_run(best_run_id)
    
    # Register the model
    model_uri = f"runs:/{best_run_id}/model"
    try:
        model_details = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        # Add tags and description if provided
        if tags or description:
            if tags:
                for key, value in tags.items():
                    client.set_model_version_tag(
                        name=model_name,
                        version=model_details.version,
                        key=key,
                        value=value
                    )
            
            if description:
                client.update_model_version(
                    name=model_name,
                    version=model_details.version,
                    description=description
                )
        
        print(f"Model registered successfully as '{model_name}' version {model_details.version}")
        return model_name, model_details.version, run
    
    except Exception as e:
        print(f"Error registering model: {e}")
        return None, None, run