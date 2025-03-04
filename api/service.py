# api/service.py

import pandas as pd
import logging
import os
import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.models.predict_model import load_model, predict_price, recommend_cars
from src.data.data_processing import load_data, preprocess_data

# Configure logging
logger = logging.getLogger(__name__)


class CarRecommendationService:
    """Service layer for car recommendations and price predictions."""
    
    def __init__(self):
        """Initialize the service, loading the model and data."""
        self.model = None
        self.car_data = None
        self.model_name = os.getenv("MODEL_NAME", "best_overall_model")
        self.model_stage = os.getenv("MODEL_STAGE", "Production")
        self.data_path = os.getenv("DATA_PATH", "data/processed/all_cars.csv")
        
        # Load model and data
        self._load_model()
        self._load_data()
    
    def _load_model(self):
        """Load the model from MLflow Model Registry."""
        try:
            logger.info(f"Loading model {self.model_name} (stage: {self.model_stage})")
            self.model = load_model(model_name=self.model_name, stage=self.model_stage)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Attempting to load the latest local model...")
            try:
                # Fallback to latest local model
                from pathlib import Path
                from src.config import MODELS_DIR
                model_files = list(MODELS_DIR.glob("*_model.joblib"))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    logger.info(f"Loading local model: {latest_model}")
                    self.model = load_model(model_path=latest_model)
                    logger.info("Local model loaded successfully")
                else:
                    logger.error("No local model files found")
            except Exception as local_err:
                logger.error(f"Error loading local model: {local_err}")
                raise
    
    def _load_data(self):
        """Load the car data for recommendations."""
        try:
            logger.info(f"Loading car data from {self.data_path}")
            self.car_data = pd.read_csv(self.data_path)
            logger.info(f"Car data loaded successfully: {len(self.car_data)} records")
        except Exception as e:
            logger.error(f"Error loading car data: {e}")
            logger.warning("Attempting to load and process raw data...")
            try:
                # Fallback to raw data
                from src.data.data_processing import load_data, preprocess_data
                raw_data = load_data()
                if raw_data is not None:
                    self.car_data = preprocess_data(raw_data)
                    logger.info(f"Raw data processed successfully: {len(self.car_data)} records")
                else:
                    logger.error("Failed to load raw data")
            except Exception as raw_err:
                logger.error(f"Error processing raw data: {raw_err}")
                raise
    
    def predict_car_price(self, car_features: Dict) -> Tuple[float, Dict]:
        """
        Predict the price of a car based on its features.
        
        Parameters:
        -----------
        car_features : Dict
            Car features
        
        Returns:
        --------
        Tuple[float, Dict]
            Predicted price and confidence interval
        """
        if self.model is None:
            logger.error("Model not loaded")
            raise ValueError("Model not loaded")
        
        try:
            # Extract brand from name
            if 'name' in car_features and 'brand' not in car_features:
                car_features['brand'] = car_features['name'].split()[0]
            
            # Create a DataFrame from the car features
            car_df = pd.DataFrame([car_features])
            
            # Extract numeric values from columns with units
            for col, unit_col in [
                ('mileage', 'mileage_numeric'),
                ('engine', 'engine_numeric'),
                ('max_power', 'max_power_numeric')
            ]:
                if col in car_df.columns:
                    from src.data.data_processing import extract_numeric_value
                    car_df[unit_col] = car_df[col].apply(extract_numeric_value)
            
            # Calculate car age
            if 'year' in car_df.columns:
                car_df['car_age'] = datetime.datetime.now().year - car_df['year']
            
            # Make prediction
            predicted_price = predict_price(self.model, car_df)
            
            # Create simple confidence interval (±10%)
            confidence_interval = {
                "lower": predicted_price * 0.9,
                "upper": predicted_price * 1.1
            }
            
            return predicted_price, confidence_interval
            
        except Exception as e:
            logger.error(f"Error predicting price: {e}")
            raise
    
    def get_car_recommendations(self, user_preferences: Dict) -> List[Dict]:
        """
        Get car recommendations based on user preferences.
        
        Parameters:
        -----------
        user_preferences : Dict
            User preferences for car recommendations
        
        Returns:
        --------
        List[Dict]
            List of recommended cars
        """
        if self.car_data is None:
            logger.error("Car data not loaded")
            raise ValueError("Car data not loaded")
        
        try:
            # Apply additional filters
            filtered_data = self.car_data.copy()
            
            if 'max_km_driven' in user_preferences and user_preferences['max_km_driven']:
                filtered_data = filtered_data[
                    filtered_data['km_driven'] <= user_preferences['max_km_driven']
                ]
            
            if 'min_year' in user_preferences and user_preferences['min_year']:
                filtered_data = filtered_data[
                    filtered_data['year'] >= user_preferences['min_year']
                ]
            
            # Get number of recommendations to return
            top_n = user_preferences.get('top_n', 5)
            
            # Use the recommendation function
            recommendations = recommend_cars(
                self.model, user_preferences, filtered_data, top_n=top_n
            )
            
            # Calculate additional helpful fields
            if len(recommendations) > 0:
                # Calculate price difference if max_price is provided
                if 'max_price' in user_preferences and user_preferences['max_price']:
                    recommendations['price_difference'] = (
                        recommendations['selling_price'] - user_preferences['max_price']
                    )
                
                # Calculate car age
                recommendations['car_age'] = datetime.datetime.now().year - recommendations['year']
                
            # Convert to list of dicts
            return recommendations.to_dict(orient='records')
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise
