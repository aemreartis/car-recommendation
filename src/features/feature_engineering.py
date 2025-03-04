# src/features/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

from ..config import NUM_FEATURES, CAT_FEATURES, TARGET_COLUMN

# Configure logging
logger = logging.getLogger(__name__)

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Create additional features from existing ones.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features
            
        Returns:
        --------
        pandas.DataFrame
            Transformed DataFrame with additional features
        """
        # Create a copy to avoid modifying the original
        X_transformed = X.copy()
        
        # Power-to-engine ratio (if both columns exist)
        if 'max_power_numeric' in X_transformed.columns and 'engine_numeric' in X_transformed.columns:
            X_transformed['power_to_engine_ratio'] = (
                X_transformed['max_power_numeric'] / 
                (X_transformed['engine_numeric'] + 1)  # Add 1 to avoid division by zero
            )
        
        # Price per year (age-adjusted price)
        if 'year' in X_transformed.columns and TARGET_COLUMN in X_transformed.columns:
            X_transformed['price_per_year'] = (
                X_transformed[TARGET_COLUMN] / 
                (2023 - X_transformed['year'] + 1)  # Add 1 to avoid division by zero
            )
        
        # Price per km (if both columns exist)
        if 'km_driven' in X_transformed.columns and TARGET_COLUMN in X_transformed.columns:
            X_transformed['price_per_km'] = (
                X_transformed[TARGET_COLUMN] / 
                (X_transformed['km_driven'] + 1)  # Add 1 to avoid division by zero
            )
        
        # Car value retention ratio (if needed columns exist)
        if 'year' in X_transformed.columns and 'km_driven' in X_transformed.columns:
            current_year = 2023
            age = current_year - X_transformed['year']
            expected_km = age * 15000  # Assuming average 15,000 km per year
            X_transformed['usage_ratio'] = X_transformed['km_driven'] / (expected_km + 1)
        
        # Brand premium (if 'brand' exists)
        if 'brand' in X_transformed.columns and TARGET_COLUMN in X_transformed.columns:
            # Calculate average price by brand
            brand_avg_price = X_transformed.groupby('brand')[TARGET_COLUMN].transform('mean')
            overall_avg_price = X_transformed[TARGET_COLUMN].mean()
            
            # Brand premium ratio
            X_transformed['brand_premium'] = brand_avg_price / overall_avg_price
        
        logger.info(f"Feature engineering completed. New shape: {X_transformed.shape}")
        return X_transformed

def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline with:
    - Feature selection
    - Missing value imputation
    - Scaling numeric features
    - One-hot encoding categorical features
    
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Preprocessing pipeline
    """
    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_FEATURES),
            ('cat', categorical_transformer, CAT_FEATURES)
        ],
        remainder='drop'  # Drop other columns
    )
    
    # Create full pipeline with custom feature engineering
    pipeline = Pipeline(steps=[
        ('feature_engineering', FeatureEngineering()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline

def prepare_features_targets(df, target_column=TARGET_COLUMN):
    """
    Split dataframe into features and target.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to split
    target_column : str
        The name of the target column
    
    Returns:
    --------
    tuple
        (X, y) where X is a DataFrame of features and y is a Series of targets
    """
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in dataframe.")
        return None, None
    
    # Features are all columns except the target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y
