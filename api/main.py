# api/main.py

import logging
import time
from typing import Dict, List
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import mlflow

from .schema import (
    CarFeatures, 
    PricePredictionResponse, 
    UserPreferences,
    CarRecommendation,
    RecommendationResponse,
    ErrorResponse
)
from .service import CarRecommendationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Car Recommendation API",
    description="API for car price prediction and recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global service instance
service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global service
    logger.info("Initializing Car Recommendation Service...")
    try:
        service = CarRecommendationService()
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing service: {e}")
        # We'll initialize lazily if this fails

# Dependency to get service
def get_service():
    """
    Get the car recommendation service.
    
    Returns:
    --------
    CarRecommendationService
        Car recommendation service instance
    """
    global service
    if service is None:
        logger.info("Initializing Car Recommendation Service (lazy)...")
        service = CarRecommendationService()
    return service

# Exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    logger.error(f"Uncaught exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(
            ErrorResponse(
                error="Internal server error",
                detail=str(exc)
            )
        )
    )

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint to check API availability."""
    return {"message": "Car Recommendation API is running"}

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check(service: CarRecommendationService = Depends(get_service)):
    """Health check endpoint to ensure all components are working."""
    health_status = {
        "status": "healthy",
        "components": {
            "api": "healthy",
            "model": "unknown",
            "data": "unknown"
        }
    }
    
    # Check model
    if service.model is not None:
        health_status["components"]["model"] = "healthy"
    else:
        health_status["components"]["model"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check data
    if service.car_data is not None and len(service.car_data) > 0:
        health_status["components"]["data"] = "healthy"
    else:
        health_status["components"]["data"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Return appropriate status code
    if health_status["status"] == "healthy":
        return health_status
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )

# Prediction endpoint
@app.post(
    "/predict", 
    response_model=PricePredictionResponse,
    tags=["Prediction"],
    summary="Predict car price",
    description="Predict the selling price of a car based on its features"
)
async def predict_car_price(
    car_features: CarFeatures,
    service: CarRecommendationService = Depends(get_service)
):
    """
    Predict the selling price of a car based on its features.
    
    Parameters:
    -----------
    car_features : CarFeatures
        Car features for price prediction
    
    Returns:
    --------
    PricePredictionResponse
        Predicted price and confidence interval
    """
    try:
        # Convert pydantic model to dict
        car_dict = car_features.dict()
        
        # Get prediction
        predicted_price, confidence_interval = service.predict_car_price(car_dict)
        
        # Return response
        return PricePredictionResponse(
            predicted_price=predicted_price,
            car_name=car_features.name,
            confidence_interval=confidence_interval
        )
    except Exception as e:
        logger.error(f"Error predicting price: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Recommendation endpoint
@app.post(
    "/recommend", 
    response_model=RecommendationResponse,
    tags=["Recommendation"],
    summary="Get car recommendations",
    description="Get car recommendations based on user preferences"
)
async def get_car_recommendations(
    preferences: UserPreferences,
    service: CarRecommendationService = Depends(get_service)
):
    """
    Get car recommendations based on user preferences.
    
    Parameters:
    -----------
    preferences : UserPreferences
        User preferences for car recommendations
    
    Returns:
    --------
    RecommendationResponse
        List of recommended cars
    """
    try:
        # Convert pydantic model to dict
        preferences_dict = preferences.dict()
        
        # Get recommendations
        recommendations = service.get_car_recommendations(preferences_dict)
        
        # Check if any recommendations were found
        if not recommendations:
            logger.warning("No recommendations found for the given preferences")
            return RecommendationResponse(
                recommendations=[],
                count=0,
                user_preferences=preferences
            )
        
        # Convert to CarRecommendation objects
        car_recommendations = [
            CarRecommendation(**recommendation)
            for recommendation in recommendations
        ]
        
        # Return response
        return RecommendationResponse(
            recommendations=car_recommendations,
            count=len(car_recommendations),
            user_preferences=preferences
        )
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# MLflow model info endpoint
@app.get(
    "/model-info",
    tags=["Model"],
    summary="Get model information",
    description="Get information about the currently loaded model"
)
async def get_model_info(service: CarRecommendationService = Depends(get_service)):
    """
    Get information about the currently loaded model.
    
    Returns:
    --------
    Dict
        Model information
    """
    try:
        # Check if model is loaded
        if service.model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not loaded"
            )
        
        # Get model info from MLflow
        model_info = {
            "model_name": service.model_name,
            "model_stage": service.model_stage,
            "pipeline_steps": [
                step[0] for step in service.model.steps
            ]
        }
        
        # Get model metadata from MLflow if possible
        try:
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(f"name='{service.model_name}'")
            
            if model_versions:
                latest_version = model_versions[0]
                model_info.update({
                    "version": latest_version.version,
                    "creation_timestamp": latest_version.creation_timestamp,
                    "run_id": latest_version.run_id
                })
        except Exception as mlflow_err:
            logger.warning(f"Could not get MLflow model metadata: {mlflow_err}")
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
