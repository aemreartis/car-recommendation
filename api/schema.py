# api/schema.py

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
import datetime

class CarFeatures(BaseModel):
    """Car features for price prediction."""
    
    name: str = Field(..., description="Full name of the car")
    year: int = Field(..., description="Year of manufacture", ge=1900, le=datetime.datetime.now().year)
    km_driven: float = Field(..., description="Total kilometers driven", ge=0)
    fuel: str = Field(..., description="Fuel type (Petrol, Diesel, CNG, LPG, Electric)")
    seller_type: str = Field(..., description="Seller type (Individual, Dealer, Trustmark Dealer)")
    transmission: str = Field(..., description="Transmission type (Manual, Automatic)")
    owner: str = Field(..., description="Ownership history (First Owner, Second Owner, etc.)")
    mileage: str = Field(..., description="Mileage with units (e.g., '23.4 kmpl')")
    engine: str = Field(..., description="Engine capacity with units (e.g., '1248 CC')")
    max_power: str = Field(..., description="Maximum power with units (e.g., '74 bhp')")
    torque: Optional[str] = Field(None, description="Torque with units (e.g., '190Nm@ 2000rpm')")
    seats: float = Field(..., description="Number of seats", ge=1, le=10)

    class Config:
        schema_extra = {
            "example": {
                "name": "Maruti Swift Dzire VDI",
                "year": 2014,
                "km_driven": 145500,
                "fuel": "Diesel",
                "seller_type": "Individual",
                "transmission": "Manual",
                "owner": "First Owner",
                "mileage": "23.4 kmpl",
                "engine": "1248 CC",
                "max_power": "74 bhp",
                "torque": "190Nm@ 2000rpm",
                "seats": 5
            }
        }

class PricePredictionResponse(BaseModel):
    """Response schema for price prediction."""
    
    predicted_price: float = Field(..., description="Predicted selling price of the car")
    car_name: str = Field(..., description="Name of the car")
    confidence_interval: Optional[Dict[str, float]] = Field(
        None, description="Confidence interval for the prediction (if available)"
    )

class UserPreferences(BaseModel):
    """User preferences for car recommendations."""
    
    max_price: Optional[float] = Field(None, description="Maximum budget", ge=0)
    brand: Optional[str] = Field(None, description="Preferred brand")
    fuel: Optional[str] = Field(None, description="Preferred fuel type")
    transmission: Optional[str] = Field(None, description="Preferred transmission type")
    max_km_driven: Optional[float] = Field(None, description="Maximum kilometers driven", ge=0)
    min_year: Optional[int] = Field(
        None, 
        description="Minimum year of manufacture", 
        ge=1900, 
        le=datetime.datetime.now().year
    )
    top_n: Optional[int] = Field(5, description="Number of recommendations to return", ge=1, le=50)

    class Config:
        schema_extra = {
            "example": {
                "max_price": 400000,
                "brand": "Maruti",
                "fuel": "Diesel",
                "transmission": "Manual",
                "max_km_driven": 150000,
                "min_year": 2012,
                "top_n": 5
            }
        }

class CarRecommendation(BaseModel):
    """Schema for a single car recommendation."""
    
    name: str
    year: int
    selling_price: float
    km_driven: float
    fuel: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    seats: float
    
    # Additional helpful fields
    price_difference: Optional[float] = None  # Difference from user's budget
    car_age: Optional[int] = None  # Age of the car in years
    
class RecommendationResponse(BaseModel):
    """Response schema for car recommendations."""
    
    recommendations: List[CarRecommendation]
    count: int
    user_preferences: UserPreferences
    
class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str
    detail: Optional[str] = None
