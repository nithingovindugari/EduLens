import sys, os
from dataclasses import dataclass
import pandas as pd
import numpy as np
import dill
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from ..logger import logging as logger
from ..exception import CustomException
from ..utils import save_object, load_object
    
@dataclass
class DataTransformationConfig:
    """
    Data class for storing data transformation configuration.

    Attributes:
        preprocessor_obj_file_path (str): File path for saving the preprocessing object.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    """
    A class to handle data transformations for machine learning preprocessing.

    Attributes:
        data_transformation_config (DataTransformationConfig): Configuration for data transformation.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a data transformer object.

        This method sets up pipelines for both numerical and categorical data processing,
        combining them into a single ColumnTransformer. This transformer can then be used
        to preprocess data for machine learning models.

        Returns:
            ColumnTransformer: A preprocessor object for transforming data.
        """
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", "race_ethnicity", "parental_level_of_education",
                "lunch", "test_preparation_course"
            ]

            # Pipeline for numerical data
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Pipeline for categorical data
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")

            # Combining both pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process for both training and testing datasets.

        This method involves reading data, applying preprocessing, and combining input features
        with target features into a single array for training and testing.

        Args:
            train_path (str): File path for the training dataset.
            test_path (str): File path for the testing dataset.

        Returns:
            tuple: A tuple containing transformed training data, testing data, and the path to the saved preprocessor object.
        """
        try:
            # Reading training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Read train and test data completed")
            logger.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"

            def prepare_features(df, target_column):
                """
                Separates input features and target feature from the dataset.

                Args:
                    df (DataFrame): The dataset from which features need to be separated.
                    target_column (str): The name of the target column.

                Returns:
                    tuple: A tuple containing input features and target feature as separate DataFrames.
                """
                input_features = df.drop(columns=[target_column], axis=1)
                target_features = df[target_column]
                return input_features, target_features

            # Preparing features for training and testing data
            input_feature_train_df, target_feature_train_df = prepare_features(train_df, target_column_name)
            input_feature_test_df, target_feature_test_df = prepare_features(test_df, target_column_name)

            logger.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Applying the preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combining input features with target features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)
