# src/models/predict_model.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import logging
from pathlib import Path

from ..config import MLFLOW_TRACKING_URI, MODELS_DIR

# Configure logging
logger = logging.getLogger(__name__)

def load_model(model_path=None, model_name=None, stage='Production'):
    """
    Load a model either from a local file or from MLflow Model Registry.
    
    Parameters:
    -----------
    model_path : str or Path, optional
        Path to a saved model file
    model_name : str, optional
        Name of the registered model in MLflow Model Registry
    stage : str, optional
        Stage of the model in MLflow Model Registry
    
    Returns:
    --------
    object
        Loaded model
    """
    if model_path is not None:
        # Load from local file
        logger.info(f"Loading model from {model_path}")
        return joblib.load(model_path)
    
    elif model_name is not None:
        # Load from MLflow Model Registry
        logger.info(f"Loading model {model_name} (stage: {stage}) from MLflow Model Registry")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        return mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    
    else:
        raise ValueError("Either model_path or model_name must be provided")

def predict_price(model, car_features):
    """
    Predict the price of a car based on its features.
    
    Parameters:
    -----------
    model : object
        Trained model
    car_features : dict or DataFrame
        Car features
    
    Returns:
    --------
    float
        Predicted price
    """
    if isinstance(car_features, dict):
        car_features = pd.DataFrame([car_features])
    
    # Make prediction
    predicted_price = model.predict(car_features)[0]
    
    return float(predicted_price)

def recommend_cars(model, user_preferences, car_data, top_n=5):
    """
    Recommend cars based on user preferences and similarity.
    
    Parameters:
    -----------
    model : object
        Trained model
    user_preferences : dict
        User preferences (e.g., preferred brand, max price, etc.)
    car_data : DataFrame
        DataFrame containing all available cars
    top_n : int, optional
        Number of recommendations to return
    
    Returns:
    --------
    DataFrame
        Recommended cars
    """
    # Create a copy of data to avoid modifying the original
    df = car_data.copy()
    
    # Apply basic filters from user preferences
    if 'max_price' in user_preferences and user_preferences['max_price']:
        df = df[df['selling_price'] <= user_preferences['max_price']]
    
    if 'brand' in user_preferences and user_preferences['brand']:
        df = df[df['brand'] == user_preferences['brand']]
    
    if 'fuel' in user_preferences and user_preferences['fuel']:
        df = df[df['fuel'] == user_preferences['fuel']]
    
    if 'transmission' in user_preferences and user_preferences['transmission']:
        df = df[df['transmission'] == user_preferences['transmission']]
    
    # If no cars match the criteria, relax constraints
    if len(df) == 0:
        logger.warning("No cars match the criteria. Relaxing constraints...")
        df = car_data.copy()
        
        # Just filter by max price if provided
        if 'max_price' in user_preferences and user_preferences['max_price']:
            df = df[df['selling_price'] <= user_preferences['max_price'] * 1.2]  # Allow 20% over budget
    
    # If still no matches, return empty DataFrame with a message
    if len(df) == 0:
        logger.warning("No cars match the relaxed criteria.")
        return pd.DataFrame(columns=car_data.columns)
    
    # Calculate a compatibility score for each car
    df['compatibility_score'] = 0
    
    # Add score based on price closeness to budget (if max_price is provided)
    if 'max_price' in user_preferences and user_preferences['max_price']:
        max_price = user_preferences['max_price']
        # Higher score for cars closer to budget
        df['price_score'] = 1 - (df['selling_price'] / max_price)
        # Cap at 0 (don't penalize cars below budget)
        df['price_score'] = df['price_score'].clip(lower=0)
        df['compatibility_score'] += df['price_score']
    
    # Add score for newer cars
    df['year_score'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    df['compatibility_score'] += df['year_score']
    
    # Add score for lower mileage
    df['mileage_score'] = 1 - (df['km_driven'] - df['km_driven'].min()) / (df['km_driven'].max() - df['km_driven'].min())
    df['compatibility_score'] += df['mileage_score']
    
    # Sort by compatibility score and return top N
    recommendations = df.sort_values('compatibility_score', ascending=False).head(top_n)
    
    # Return without the added score columns
    score_columns = ['compatibility_score', 'price_score', 'year_score', 'mileage_score']
    return recommendations.drop(columns=[col for col in score_columns if col in recommendations.columns])
