import os
import sys

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:  # wb is write byte mode
            dill.dump(obj, file_obj)
        pass
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """Function to fetch r2_score of models

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
        models (_type_): _description_

    Returns:
        dict: (key, val) = (model_names, r2_score)
    """
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            parameters_for_model_i = params[list(models.keys())[i]]

            gs = GridSearchCV(model, parameters_for_model_i, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
