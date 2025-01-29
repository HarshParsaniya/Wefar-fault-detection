import os
import sys

from wefar_fault_detection.components.supervised.data_transformation import DataTransformation
from wefar_fault_detection.components.supervised.model_trainer import ModelTrainer


if __name__ == '__main__':
    data_transformation = DataTransformation()
    train_df, test_df = data_transformation.initiate_data_transformation()
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_df, test_df)