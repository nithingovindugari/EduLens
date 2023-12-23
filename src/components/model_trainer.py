import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from ..logger import logging as logger
from ..exception import CustomException
from ..utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """
    Configuration for the ModelTrainer class.
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    """
    A class to train different regression models and select the best one based on performance.
    """

    def __init__(self):
        """
        Initialize the ModelTrainer with a configuration.
        """
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple models and selects the best model based on R2 score.

        Args:
            train_array (numpy.ndarray): Training data.
            test_array (numpy.ndarray): Testing data.

        Returns:
            float: The R2 score of the best model.

        Raises:
            CustomException: If no model meets the performance threshold.
        """
        try:
            logger.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Model configurations
            models, params = self.get_models_and_params()

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_name, best_model, best_model_score = self.select_best_model(model_report, models)

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logger.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            self.save_best_model(best_model)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

    def get_models_and_params(self):
        """
        Defines the models and their parameter grids for training.

        Returns:
            tuple: A tuple containing two dictionaries, one for models and another for parameters.
        """
        models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
        }
        params = {
            "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
            "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
            "Gradient Boosting": {
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Linear Regression": {},
            "XGBRegressor": {
                'learning_rate': [.1, .01, .05, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "CatBoosting Regressor": {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            },
            "AdaBoost Regressor": {
                'learning_rate': [.1, .01, 0.5, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }
        return models, params

    def select_best_model(self, model_report, models):
        """
        Selects the best model based on the evaluation report.

        Args:
            model_report (dict): A dictionary containing model performance metrics.
            models (dict): A dictionary of trained model instances.

        Returns:
            tuple: A tuple containing the name, instance, and score of the best model.
        """
        best_model_score = max(model_report.values())
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        return best_model_name, best_model, best_model_score

    def save_best_model(self, model):
        """
        Saves the best model to a file.

        Args:
            model (sklearn.base.BaseEstimator): The model to be saved.
        """
        save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=model)
        
        