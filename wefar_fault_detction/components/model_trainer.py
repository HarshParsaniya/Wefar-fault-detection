import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import mlflow
from mlflow.sklearn import log_model

from wefar_fault_detction.exception.exception import CustomException
from wefar_fault_detction.utils.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    trained_model_grid_search_file_path = os.path.join('artifacts', 'model_grid_search.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def intitate_model_training(self, train_array, test_array):
        try:
            with mlflow.start_run(run_name="Model Training"):

                # Split train and test arrays
                X_train, X_test, y_train, y_test = (
                    train_array[:, :-1],
                    test_array[:, :-1],
                    train_array[:, -1],
                    test_array[:, -1]
                )

                # Initialize and train the base model
                model = LogisticRegression()
                model.fit(X_train, y_train)

                # Evaluate the model
                model_report = evaluate_model(X_train, X_test, y_train, y_test, model)
                mlflow.log_metrics({
                    "Base Accuracy": model_report["accuracy Score"],
                    "Base ROC AUC": model_report["roc_auc Score"]
                })

                # Define parameter grid for GridSearchCV
                param_grid = [
                    {
                        'penalty': ['l2'],
                        'solver': ['lbfgs'],
                        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                        'class_weight': [None, 'balanced'],
                        'max_iter': [500, 1000, 1500]
                    },
                    {
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear'],
                        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                        'class_weight': [None, 'balanced']
                    }
                ]

                # Train model with GridSearchCV
                grid_search = GridSearchCV(
                    estimator=LogisticRegression(),
                    param_grid=param_grid,
                    scoring=['accuracy', 'f1_weighted', 'roc_auc'],
                    refit='f1_weighted',
                    cv=5,
                    n_jobs=-1,
                    verbose=2,
                    error_score='raise'
                )
                grid_search.fit(X_train, y_train)

                # Log best parameters and scores
                best_model_parameters = grid_search.best_params_
                mlflow.log_params(best_model_parameters)

                # Evaluate the best model
                best_model_report = evaluate_model(X_train, X_test, y_train, y_test, grid_search)
                mlflow.log_metrics({
                    "Best Accuracy": best_model_report["accuracy Score"],
                    "Best ROC AUC": best_model_report["roc_auc Score"]
                })

                # Save models and log as artifacts
                save_object(self.model_trainer_config.trained_model_file_path, model)
                save_object(self.model_trainer_config.trained_model_grid_search_file_path, grid_search)
                log_model(model, "Base Logistic Regression Model")
                log_model(grid_search, "Optimized Logistic Regression Model")


        except Exception as e:
            mlflow.log_metric("training_status", 0)
            raise CustomException(e, sys) from e