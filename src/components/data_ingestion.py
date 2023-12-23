import os
import sys
from ..exception import CustomException
from ..logger import logging as logger
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from .data_transformation import DataTransformation
from .data_transformation import DataTransformationConfig

from .model_trainer import ModelTrainerConfig
from .model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    Data class for storing configuration paths related to data ingestion.

    Attributes:
        train_data_path (str): Path to save the training data.
        test_data_path (str): Path to save the testing data.
        raw_data_path (str): Path to save the raw data.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    """
    A class to handle the ingestion of data for a machine learning workflow.

    This class reads data, splits it into training and testing sets, and saves them.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiates the process of data ingestion.

        This method involves reading a dataset, performing a train-test split,
        and saving the splits into designated paths.

        Returns:
            tuple: Paths where the training and testing datasets are saved.
        """
        logger.info("Entered the data ingestion method or component")
        try:
            # Read the dataset
            df = pd.read_csv("Notebooks/Data/stud.csv")
            logger.info('Read the dataset as dataframe')

            # Ensure the directory for saving data exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logger.info("Train test split initiated")
            # Perform train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
    