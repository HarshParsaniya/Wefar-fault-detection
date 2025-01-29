import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from wefar_fault_detection.exception.exception import CustomException
from wefar_fault_detection.utils.utils import save_object

import mlflow



@dataclass
class ModelTrainerConfig:
    logisticregression_data_path = os.path.join('artifacts/supervised/models', 'logreg.pkl')
    decisiontreeclassifier_data_path = os.path.join('artifacts/supervised/models', 'decision_tree.pkl')
    svc_data_path = os.path.join('artifacts/supervised/models', 'svc.pkl')
    randomforestclassifier_data_path = os.path.join('artifacts/supervised/models', 'rfc.pkl')
    xgbclassifier_data_path = os.path.join('artifacts/supervised/models', 'xgboost.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            with mlflow.start_run(run_name="Model Training"):
                # Splitting Dependent and Independent variables from train and test data
                X_train, X_test, y_train, y_test = (
                    train_array[:, :-1],
                    test_array[:, :-1],
                    train_array[:, -1],
                    test_array[:, -1]
                )

                mlflow.log_param('Training_Row_shape', X_train.shape[0])
                mlflow.log_param('Training_Column_shape', X_train.shape[1])
                mlflow.log_param('Testing_Row_shape', X_test.shape[0])
                mlflow.log_param('Testing_Column_shape', X_test.shape[1])

                os.makedirs(os.path.dirname(self.model_trainer_config.logisticregression_data_path), exist_ok=True)

                # Create a model list
                models = {
                    "LogisticRegression": LogisticRegression(max_iter=10000),
                    "DecisionTreeClassifier": DecisionTreeClassifier(),
                    "SVC": SVC(probability=True),
                    "RandomForestClassifier": RandomForestClassifier(n_estimators=100),
                    "XGBClassifier": XGBClassifier()
                }
                # Check for NaN values
                print(f"NaN values in y_train: {np.isnan(y_train).sum()}")
                print(f"NaN values in y_test: {np.isnan(y_test).sum()}")

                all_model_report = {}
                for i in range(len(list(models))):
                    model = list(models.values())[i]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    all_model_report[list(models.keys())[i]] = accuracy

                    # Save the model
                    model_path = getattr(self.model_trainer_config, f"{list(models.keys())[i].lower()}_data_path")
                    save_object(model_path, model)
                    mlflow.log_param(f"{list(models.keys())[i]}_model_path", model_path)

                # Log accuracy scores
                mlflow.log_dict(all_model_report, "model_accuracy.json")

                mlflow.log_param('Accuracy_score', all_model_report)

                # Select the best model
                best_model_name = max(all_model_report, key=all_model_report.get)
                # best_model_name = max(all_model_report.keys(), key=lambda key: all_model_report[key])     # equivalent to the above
                best_model_score = max(list(all_model_report.values()))

                mlflow.log_param('Best_Model_Name', best_model_name)
                mlflow.log_metric('Best_Model_Score', best_model_score)

        
        except Exception as e:
            raise CustomException(e, sys)