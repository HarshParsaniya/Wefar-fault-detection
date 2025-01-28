import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import mlflow
from mlflow.sklearn import log_model

from wefar_fault_detection.exception.exception import CustomException
from wefar_fault_detection.utils.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def intitate_model_training(self, train_array, test_array):
        try:
            with mlflow.start_run(run_name="Model Training"):
                train_model_report = []
                test_model_report = []
                best_model = None
                final_k = 0

                for k in range(2, 11):
                    # Initialize and train the base model
                    model = AgglomerativeClustering(n_clusters=k)
                    model.fit(train_array)

                    # Evaluate the train model
                    train_score = silhouette_score(train_array, model.labels_)
                    train_model_report.append(train_score)

                    # Evaluate the test model
                    test_labels = model.fit_predict(test_array)
                    test_score = silhouette_score(test_array, test_labels)
                    test_model_report.append(test_score)

                    # Track the best model based on train silhouette score
                    if train_score > max(train_model_report):
                        final_k = k
                        best_model = model
                    

                mlflow.log_metrics({
                    "Accuracy": train_model_report[final_k-2]
                })

                mlflow.log_params({
                    "Accuracy": test_model_report[final_k-2]
                })

                # Save models and log as artifacts
                save_object(self.model_trainer_config.trained_model_file_path, best_model)
                log_model(best_model, "Hierarchical cluster Model")

                mlflow.log_metric('Training status', 1)


        except Exception as e:
            mlflow.log_metric("training_status", 0)
            raise CustomException(e, sys) from e