import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import mlflow
from wefar_fault_detction.exception.exception import CustomException
from wefar_fault_detction.utils.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
        self.target_column_name = 'Income'

    def get_data_transformation_obj(self):
        try:
            categorical_cols = ['Workclass', 'Occupation', 'Native Country']
            numerical_cols = ['Age', 'Final Weight', 'EducationNum', 'Capital Gain', 'capital loss', 'Hours per Week']

            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
            ])

            preprocessor = ColumnTransformer([
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ])

            # Log preprocessing steps in MLflow
            mlflow.log_param("numerical_columns", numerical_cols)
            mlflow.log_param("categorical_columns", categorical_cols)

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            with mlflow.start_run(run_name="Data Transformation"):
                # Load train and test datasets
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                mlflow.log_param("train_data_shape", train_df.shape)
                mlflow.log_param("test_data_shape", test_df.shape)

                # Get preprocessor object
                preprocessor = self.get_data_transformation_obj()

                # Separate features and target
                X_train, y_train = train_df.drop(self.target_column_name, axis=1), train_df[self.target_column_name]
                X_test, y_test = test_df.drop(self.target_column_name, axis=1), test_df[self.target_column_name]

                # Log target distribution
                mlflow.log_metric("train_target_0", (y_train == 0).sum())
                mlflow.log_metric("train_target_1", (y_train == 1).sum())
                mlflow.log_metric("test_target_0", (y_test == 0).sum())
                mlflow.log_metric("test_target_1", (y_test == 1).sum())

                # Transform features
                X_train_transformed = preprocessor.fit_transform(X_train)
                X_test_transformed = preprocessor.transform(X_test)

                # Combine transformed features with target
                train_data = np.c_[X_train_transformed, y_train]
                test_data = np.c_[X_test_transformed, y_test]

                # Save the preprocessor object
                save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
                mlflow.log_artifact(self.data_transformation_config.preprocessor_obj_file_path, artifact_path="preprocessors")

                # Log transformation results
                mlflow.log_param("transformed_train_shape", train_data.shape)
                mlflow.log_param("transformed_test_shape", test_data.shape)

                # Indicate successful transformation
                mlflow.log_metric("transformation_status", 1)

                return train_data, test_data, self.data_transformation_config.preprocessor_obj_file_path
            
        except Exception as e:
            # Log failure and error details in MLflow
            mlflow.log_metric("transformation_status", 0)
            error_message = str(e)

            with open("transformation_error.log", "w") as error_file:
                error_file.write(error_message)

            mlflow.log_artifact("transformation_error.log", artifact_path="errors")
            
            raise CustomException(e, sys)
