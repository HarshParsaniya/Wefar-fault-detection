import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wefar_fault_detection.exception.exception import CustomException
from wefar_fault_detection.utils.utils import save_object, load_object

import mlflow



@dataclass
class MergeTargetConfig:
    train_data_path = os.path.join('artifacts/supervised', 'train.csv')
    test_data_path = os.path.join('artifacts/supervised', 'test.csv')

class MergeTarget:
    def __init__(self):
        self.merge_target_config = MergeTargetConfig()

    def initiate_merge(self):
        try:
            with mlflow.start_run(run_name='Merge Target'):
                # Read the train and test data
                train_data = pd.read_csv(os.path.join('artifacts/unsupervised', 'train.csv'))
                train_data_columns_name = train_data.columns
                test_data = pd.read_csv(os.path.join('artifacts/unsupervised', 'test.csv'))

                # Load the preprocessor and model
                preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
                model_path = os.path.join('artifacts', 'model.pkl')

                preprocessor = load_object(preprocessor_path)
                model = load_object(model_path)

                # Scaled the data using preprocessor of train data
                test_data_scaled = preprocessor.transform(test_data)

                # predict the test data
                model.fit_predict(test_data_scaled)

                test_labels = model.labels_

                # Add the target column to the test data
                test_data['Good/Bad'] = test_labels

                # make directory
                os.makedirs(os.path.dirname(self.merge_target_config.train_data_path), exist_ok=True)

                # store the train and test datasets
                train_data.to_csv(self.merge_target_config.train_data_path, index=False, header=True)
                test_data.to_csv(self.merge_target_config.test_data_path, index=False, header=True)

                mlflow.log_param('Merge the Target column on test dataset', 1)


        except Exception as e:
            mlflow.log_param('Error Merge the Target column on test dataset', 0)
            raise CustomException(e, sys)