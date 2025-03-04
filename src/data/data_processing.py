# src/data/data_processing.py

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import sys
from ..config import DATA_DIR, TEST_SIZE, VAL_SIZE, DEFAULT_RANDOM_STATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def extract_numeric_value(value, default_value=np.nan):
    """Extract numeric value from a string containing units."""
    if pd.isna(value):
        return default_value
    
    if isinstance(value, (int, float)):
        return value
        
    # Try to extract numeric value
    match = re.search(r'([\d.]+)', str(value))
    if match:
        return float(match.group(1))
    
    return default_value

def load_data(filepath=None):
    """Load car dataset and perform initial processing."""
    if filepath is None:
        filepath = DATA_DIR / "raw" / "car_data.csv"
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Clean and preprocess the dataset."""
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Extract numeric values from columns with units
    processed_df['mileage_numeric'] = processed_df['mileage'].apply(
        lambda x: extract_numeric_value(x)
    )
    processed_df['engine_numeric'] = processed_df['engine'].apply(
        lambda x: extract_numeric_value(x)
    )
    processed_df['max_power_numeric'] = processed_df['max_power'].apply(
        lambda x: extract_numeric_value(x)
    )
    
    # Extract car brand from name
    processed_df['brand'] = processed_df['name'].apply(
        lambda x: str(x).split()[0] if pd.notna(x) else np.nan
    )
    
    # Handle missing values
    for col in ['mileage_numeric', 'engine_numeric', 'max_power_numeric']:
        median_value = processed_df[col].median()
        processed_df[col].fillna(median_value, inplace=True)
    
    # Convert categorical columns to categorical data type for efficiency
    cat_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
    for col in cat_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype('category')
    
    # Calculate car age (assuming current year is 2023)
    processed_df['car_age'] = 2023 - processed_df['year']
    
    logger.info(f"Data preprocessing completed. Shape: {processed_df.shape}")
    return processed_df

def split_data(df):
    """Split data into train, validation, and test sets."""
    # First, split into train+val and test
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=DEFAULT_RANDOM_STATE
    )
    
    # Then, split train+val into train and validation
    train_df, val_df = train_test_split(
        train_val_df, test_size=VAL_SIZE, random_state=DEFAULT_RANDOM_STATE
    )
    
    logger.info(f"Train set: {train_df.shape[0]} rows")
    logger.info(f"Validation set: {val_df.shape[0]} rows")
    logger.info(f"Test set: {test_df.shape[0]} rows")
    
    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df):
    """Save the processed datasets to CSV files."""
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    
    # Save the combined data for later use
    combined_df = pd.concat([train_df, val_df, test_df])
    combined_df.to_csv(processed_dir / "all_cars.csv", index=False)
    
    logger.info(f"Datasets saved to {processed_dir}")

def prepare_data():
    """Main function to prepare the data."""
    # Load raw data
    df = load_data()
    if df is None:
        return None, None, None
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    # Split data
    train_df, val_df, test_df = split_data(processed_df)
    
    # Save processed data
    save_processed_data(train_df, val_df, test_df)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_data()
