import mlflow
import sys
from mlflow.tracking import MlflowClient

from src.models.train_model import find_best_run_and_register

if __name__ == "__main__":
    try:
        # Check connection to MLflow server
        client = MlflowClient()
        experiments = mlflow.search_experiments()
        print(f"Successfully connected to MLflow server with {len(experiments)} experiments")
    except Exception as e:
        print(f"Error connecting to MLflow server: {e}")
        print("Please ensure your MLflow server is running at http://localhost:5000")
        sys.exit(1)

    # Find best model across specific experiments
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
    