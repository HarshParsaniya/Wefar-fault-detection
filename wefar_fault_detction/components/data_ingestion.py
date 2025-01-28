import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from wefar_fault_detction.exception.exception import CustomException
import mlflow

@dataclass
class DataIngestionconfig:
    raw_data_path = os.path.join('artifacts', 'raw.csv')
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        try:
            with mlflow.start_run(run_name="Data Ingestion"):
                # Load and save the raw dataset
                df = pd.read_csv(os.path.join('notebook/data/census-income', 'adult.csv'))
                mlflow.log_param("raw_dataset_shape", df.shape)
                os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
                df.to_csv(self.ingestion_config.raw_data_path, index=False)
                mlflow.log_artifact(self.ingestion_config.raw_data_path, artifact_path="datasets/raw")

                # Preprocess the dataset
                df.drop(['Education', 'Marital Status', 'Relationship', 'Race', 'Gender'], axis=1, inplace=True)
                df.replace({'?': np.nan}, inplace=True)
                df['Income'] = df['Income'].str.strip().replace({'<=50K': 0, '>50K': 1}).astype(int)
                mlflow.log_param("processed_columns", list(df.columns))
                mlflow.log_metric("missing_values_count", df.isnull().sum().sum())

                # Split into train and test datasets
                train_dataset, test_dataset = train_test_split(df, test_size=0.3, random_state=42)
                mlflow.log_param("train_dataset_shape", train_dataset.shape)
                mlflow.log_param("test_dataset_shape", test_dataset.shape)

                # Save train and test datasets
                train_dataset.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
                test_dataset.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
                mlflow.log_artifact(self.ingestion_config.train_data_path, artifact_path="datasets/train")
                mlflow.log_artifact(self.ingestion_config.test_data_path, artifact_path="datasets/test")
                mlflow.log_metric("ingestion_status", 1)

                return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            mlflow.log_metric("ingestion_status", 0)
            raise CustomException(e, sys)