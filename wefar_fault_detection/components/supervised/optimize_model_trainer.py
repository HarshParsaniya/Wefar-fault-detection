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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from wefar_fault_detection.exception.exception import CustomException
from wefar_fault_detection.utils.utils import save_object

import mlflow




@dataclass
class OptimizeModelTrainerConfig:
    logisticregression_data_path = os.path.join('artifacts/supervised/optimize_models', 'logreg.pkl')
    decisiontreeclassifier_data_path = os.path.join('artifacts/supervised/optimize_models', 'decision_tree.pkl')
    svc_data_path = os.path.join('artifacts/supervised/optimize_models', 'svc.pkl')
    randomforestclassifier_data_path = os.path.join('artifacts/supervised/optimize_models', 'rfc.pkl')
    xgbclassifier_data_path = os.path.join('artifacts/supervised/optimize_models', 'xgboost.pkl')


class OptimizeModelTrainer:
    def __init__(self):
        self.optimize_model_trainer = OptimizeModelTrainerConfig()

    def initiate_optimize_model_trainer(self, train_array, test_array):
        try:
            with mlflow.start_run(run_name='Optimize Model Trainer'):
                # param_grid for optimize the model
                param_grid = {
                    'LogisticRegression' : {
                        'penalty': ['l1', 'l2'],
                        'C': [0.001, 0.01, 0.1, 1, 10],
                        'max_iter': [100, 500, 1000],
                        'solver': ['liblinear'],
                        'multi_class': ['auto']
                    },

                    'DecisionTreeClassifier' : {
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [3, 5, 10, 20, 50],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 5, 10],
                        'max_features': ['sqrt', 'log2']
                    },

                    'SVC' : {
                        'kernel': ['linear', 'rbf', 'poly'],
                        'C': [0.001, 0.01, 0.1, 1, 10],
                        'gamma': ['scale', 'auto'],
                        'probability': [True, False],
                    },

                    'RandomForestClassifier' : {
                        'n_estimators': [10, 50, 100, 200, 500],
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [3, 5, 10, 20, 50],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 5, 10],
                        'max_features': ['sqrt', 'log2'],
                    },

                    'XGBClassifier' : {
                        'n_estimators': [10, 50, 100, 200, 500],
                        'learning_rate': [0.01, 0.1, 0.5, 1],
                        'max_depth': [3, 5, 10, 20, 50],
                        'subsample': [0.5, 0.75, 1]
                    }
                }

                # Splitting Dependent and Independent variables from train and test data
                X_train, X_test, y_train, y_test = (
                    train_array[:, :-1],
                    test_array[:, :-1],
                    train_array[:, -1],
                    test_array[:, -1]
                )

                mlflow.log_dict(param_grid, "optimize_model_param_grid.json")
                
                mlflow.log_param('Training_Row_shape', X_train.shape[0])
                mlflow.log_param('Training_Column_shape', X_train.shape[1])
                mlflow.log_param('Testing_Row_shape', X_test.shape[0])
                mlflow.log_param('Testing_Column_shape', X_test.shape[1])

                os.makedirs(os.path.dirname(self.optimize_model_trainer.logisticregression_data_path), exist_ok=True)

                # Create a model list
                models = {
                    "LogisticRegression": LogisticRegression(),
                    "DecisionTreeClassifier": DecisionTreeClassifier(),
                    "SVC": SVC(),
                    "RandomForestClassifier": RandomForestClassifier(),
                    "XGBClassifier": XGBClassifier()
                }

                # Check for NaN values
                print(f"NaN values in y_train: {np.isnan(y_train).sum()}")
                print(f"NaN values in y_test: {np.isnan(y_test).sum()}")
                # y_train = np.nan_to_num(y_train)
                # y_test = np.nan_to_num(y_test)

                all_model_report = {}
                for model_name, model in models.items():
                    try:
                        grid_search = GridSearchCV(
                            model, 
                            param_grid=param_grid[model_name],
                            cv=5,
                            scoring='accuracy',
                            n_jobs=-1
                        )
                        grid_search.fit(X_train, y_train)
                        y_pred = grid_search.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        all_model_report[model_name] = accuracy

                        # Save the model
                        model_path = getattr(self.optimize_model_trainer, f"{model_name.lower()}_data_path")
                        save_object(model_path, grid_search)
                        mlflow.log_param(f"{model_name}_model_path", model_path)
                        mlflow.log_metric(f"{model_name}_Accuracy", accuracy)

                    except Exception as model_error:
                        print(f"Error training {model_name}: {model_error}")
                        mlflow.log_param(f"{model_name}_Error", str(model_error))

                # Log accuracy scores
                mlflow.log_dict(all_model_report, "optimize_model_accuracy.json")

                mlflow.log_param('Optimize_Accuracy_score', all_model_report)

                # Select the best model
                best_model_name = max(all_model_report, key=all_model_report.get)
                # best_model_name = max(all_model_report.keys(), key=lambda key: all_model_report[key])     # equivalent to the above
                best_model_score = max(list(all_model_report.values()))

                mlflow.log_param('Optimize_Best_Model_Name', best_model_name)
                mlflow.log_metric('Optimize_Best_Model_Score', best_model_score)

        except Exception as e:
            raise CustomException(e, sys)