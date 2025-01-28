import os
import sys

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from wefar_fault_detection.exception.exception import CustomException


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
    


def load_object(file_path):
    try:
        # load the pickle file
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)