import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from wefar_fault_detection.exception.exception import CustomException
from wefar_fault_detection.utils.utils import load_object

import mlflow



@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path = os.path.join('artifacts/supervised', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
        self.target_column = 'Good/Bad'
        self.drop_columns = [self.target_column, 'Unnamed: 0']

    def get_data_transformation_obj(self, df):
        try:
            # Differentiate numerical column and non numerical column name
            non_numerical_columns_name = df.select_dtypes(exclude=['float64']).columns

            # Differentiate numerical column and non numerical column
            df_non_numerical_columns = df.select_dtypes(exclude='float64')

            # Log preprocessing steps in MLflow
            mlflow.log_param("non_numerical_columns", list(non_numerical_columns_name))

            # Convert the date value into null value
            df_non_numerical_columns = df_non_numerical_columns.apply(lambda x: pd.to_datetime(x, format='%d-%m-%Y', errors='coerce'))

            # Convert non numerical value into float
            df_non_numerical_columns = df_non_numerical_columns.apply(pd.to_numeric, errors='coerce')
            # df_non_numerical_columns = df_non_numerical_columns.applymap(lambda x: float(x))

            # merge numerical and transform non numerical columns
            df[non_numerical_columns_name] = df_non_numerical_columns

            numerical_pieline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('train', numerical_pieline, list(df.columns))
            ])

            return preprocessor
        

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self):
        try:
            with mlflow.start_run(run_name="Data Transformation"):
                # Read the data path
                train_dataset_path = os.path.join('artifacts/supervised', 'train.csv')
                test_dataset_path = os.path.join('artifacts/supervised', 'test.csv')

                # Read the train and test data 
                train_df = pd.read_csv(train_dataset_path)
                test_df = pd.read_csv(test_dataset_path)

                mlflow.log_param("train_data_shape", train_df.shape)
                mlflow.log_param("test_data_shape", test_df.shape)

                # Remove null columns
                train_df.dropna(axis=1, how='all', inplace=True)

                # input features for train and test
                X_train = train_df.drop(self.drop_columns, axis=1, errors='ignore')
                X_test = test_df.drop(self.drop_columns, axis=1, errors='ignore')
                y_train = train_df[self.target_column]
                y_test = test_df[self.target_column]

                # Get preprocessor object
                preprocessor = self.get_data_transformation_obj(X_train)

                # Transform features
                X_train_transformed = preprocessor.fit_transform(X_train)
                X_test_transformed = preprocessor.transform(X_test)

                # Get Model object
                model_path = os.path.join('artifacts', 'model.pkl')
                model = load_object(model_path)

                # Transform target column of train data
                model.fit(X_train_transformed)

                y_train_predict_labels = model.labels_

                # Log transformation results
                mlflow.log_param("transformed_train_shape", X_train_transformed.shape)
                mlflow.log_param("transformed_test_shape", X_test_transformed.shape)

                # Indicate successful transformation
                mlflow.log_metric("transformation_status", 1)

                scaled_train = np.c_[X_train_transformed, np.array(y_train_predict_labels)]
                scaled_test = np.c_[X_test_transformed, np.array(y_test)]

                return scaled_train, scaled_test


        except Exception as e:
            # Log failure and error details in MLflow
            mlflow.log_metric("transformation_status", 0)
            error_message = str(e)

            with open("transformation_error.log", "w") as error_file:
                error_file.write(error_message)

            mlflow.log_artifact("transformation_error.log", artifact_path="errors")
            raise CustomException(e, sys)