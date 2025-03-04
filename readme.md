# Car Recommendation System

An end-to-end machine learning pipeline for car price prediction and recommendation using MLflow and FastAPI.

## Project Overview

This project implements a complete solution for:
- Car price prediction based on features
- Personalized car recommendations based on user preferences
- Model training, tracking, and deployment using MLflow
- API serving using FastAPI

## Architecture

![Architecture Diagram](architecture.png)

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

## Project Structure

```
car_recommendation/
├── api/                   # FastAPI application
│   ├── main.py            # API endpoints
│   ├── schema.py          # Pydantic models
│   └── service.py         # Service layer
├── data/                  # Data files
│   ├── raw/               # Original data
│   └── processed/         # Processed data
├── mlruns/                # MLflow experiment tracking
├── models/                # Saved models
├── src/                   # Source code
│   ├── data/              # Data processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and prediction
│   ├── utils/             # Utility functions
│   └── config.py          # Configuration
├── Dockerfile             # Dockerfile for containerization
├── docker-compose.yml     # Docker Compose configuration
├── main.py                # Main entry point
└── requirements.txt       # Python dependencies
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.8+

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd car-recommendation-system
   ```

2. Set up the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the Project

#### Using Docker (Recommended)

1. Start all services:
   ```bash
   docker-compose up
   ```

2. Access the services:
   - MLflow UI: http://localhost:5000
   - FastAPI Swagger UI: http://localhost:8000/docs

#### Manual Setup

1. Start MLflow server:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

2. Train the model:
   ```bash
   python main.py --optimize --register_model
   ```

3. Start the FastAPI service:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Usage

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

## Model Training

To train the model with different configurations:

```bash
# Basic training
python main.py

# With hyperparameter optimization
python main.py --optimize

# With hyperparameter optimization and a specific number of trials
python main.py --optimize --trials 50

# With hyperparameter optimization and model registration
python main.py --optimize --register_model
```

## Customization

### Adding New Features

1. Update the feature definitions in `src/config.py`
2. Implement the feature extraction in `src/features/feature_engineering.py`
3. Retrain the model

### Adding New Models

1. Add the model configuration in `src/config.py`
2. Update the training logic in `src/models/train_model.py`
3. Retrain and compare with existing models

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
