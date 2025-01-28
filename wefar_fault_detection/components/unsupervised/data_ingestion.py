import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wefar_fault_detection.exception.exception import CustomException

import mlflow





@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts/unsupervised', 'train.csv')
    test_data_path: str = os.path.join('artifacts/unsupervised', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            with mlflow.start_run(run_name="Data Ingestion"):
                # Read the train and test dataset from csv file
                train_data = pd.read_csv(os.path.join('notebook/dataset', 'training_dataset.csv'))
                mlflow.log_param('Shape of Training dataset', train_data.shape)

                test_data = pd.read_csv(os.path.join('notebook/dataset', 'test_dataset.csv'))
                mlflow.log_param('Shape of Test dataset', test_data.shape)

                # Create a artifacts folder
                os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
                mlflow.log_param('Created artifacts folder', 'successfully')

                # Save train and test datasets
                train_data.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
                mlflow.log_artifact(self.data_ingestion_config.train_data_path, artifact_path='train')

                test_data.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
                mlflow.log_artifact(self.data_ingestion_config.test_data_path, artifact_path='test')

                mlflow.log_metric('Ingestion status', 1)

                return self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path
        
        except Exception as e:
            mlflow.log_metric('Ingestion status', 0)
            raise CustomException(e, sys)