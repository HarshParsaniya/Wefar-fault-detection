import os
import sys

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from wefar_fault_detction.exception.exception import CustomException


def save_object(file_path, obj):
    try:
        # create path for storing preprocessor object
        dir_path = os.path.dirname(file_path)

        # create directory to store preprocessor object
        os.makedirs(dir_path, exist_ok=True)
        
        # create a pickle file
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, X_test, y_train, y_test, model):
    try:
        report = {}

        # Train model
        model.fit(X_train,y_train)

        # Predict Testing data
        y_test_pred =model.predict(X_test)

        # Get accuracy Score for train and test data
        test_model_accuracy_score = accuracy_score(y_test,y_test_pred)

        # Store the accuracy Score in report
        report['accuracy Score'] = test_model_accuracy_score

        # Get probabilities 
        y_prob = model.predict_proba(X_test)

        # Compute ROC AUC Score
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])

        # Store the roc auc score in report
        report['roc_auc Score'] = roc_auc

        return report
    

    except Exception as e:
        raise CustomException(e,sys)
    


def load_object(file_path):
    try:
        # load the pickle file
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)