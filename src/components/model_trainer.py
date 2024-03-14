import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logs import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.plk')

class modelTrainer:
    def __init__(self):
        self.model_trainer_config()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting Training and Test input data.")
            X_train, y_train, X_test, y_test =(
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'XGB Regressor': XGBRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=False),
                'AdaBoost Regressor': AdaBoostRegressor(),
            }
            
            model_report: dict = evaluate_models(X_train=X_train, y=y_train, X_test=X_test, y_test=y_test, models=models)

            # Get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            # Get best model name from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("There is not a best model. Train again.")
            logging.info(f"Best model found on Training and Test datasets.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            r2_squared = r2_score(y_test, predicted)
            return r2_squared     
        
        except Exception as e:
            raise CustomException(e, sys)