## Project Overview
This project implements a complete solution for:
- Car price prediction based on features
- Personalized car recommendations based on user preferences
- Model training, tracking, and deployment using MLflow
- API serving using FastAPI

## Architectur

The system consists of the following components:
- **Data Processing**: Cleaning and preprocessing car data
- **Feature Engineering**: Creating meaningful features from raw data
- **Model Training**: Training and comparing multiple regression models
- **MLflow Tracking**: Experiment tracking and model registry
- **FastAPI Service**: Serving predictions and recommendations via REST API

## Models
The system trains and compares the following models:
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

Models are evaluated based on RMSE, MAE, and R² metrics. The best model is selected for deployment.

### Prerequisites
- Docker and Docker Compose
- Python 3.8+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aemreartis/car-recommendation.git
   cd car-recommendation-system
   ```

### Running the Project

#### Using Docker 

1. Start MLFlow, FastAPI and Train services:
   ```bash
   docker-compose up
   ```
2. Register best model by given metric:
   ```bash
   docker compose --profile post-training up -d 
   ```
  
4. Access the services:
   - MLflow UI: http://localhost:5000
   - FastAPI Swagger UI: http://localhost:8000/docs

### API Endpoints

The API provides the following endpoints:

- `GET /`: Root endpoint, checks API availability
- `GET /health`: Health check endpoint
- `POST /predict`: Predict car price from features
- `POST /recommend`: Get car recommendations based on preferences
- `GET /model-info`: Get information about the loaded model

### Example API Requests

#### Price Prediction

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
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
}'
```

#### Car Recommendation

```bash
curl -X 'POST' \
  'http://localhost:8000/recommend' \
  -H 'Content-Type: application/json' \
  -d '{
  "max_price": 500000,
  "brand": "Maruti",
  "fuel": "Diesel",
  "transmission": "Manual",
  "max_km_driven": 150000,
  "min_year": 2010
}'
```
