import os
import sys
from ..exception import CustomException
from ..logger import logging as logger
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass  # Decorator to make the class immutable and hashable 
class DataIngestion_Path:
    train_data_path : str = os.path.join("artifacts", "train.csv")
    test_data_path : str = os.path.join("artifacts", "test.csv")
    raw_data_path : str = os.path.join("artifacts", "raw_data.csv")
    
    
class DataIngestion:
    
    # Constructor for DataIngestion class to initialize the path variables 
    def __init__(self):
        self.ingestion_path = DataIngestion_Path()
        
    
    def initiate_data_ingestion(self, data_path):
        """
        This method is used to initiate the data ingestion process.
        :param data_path: Path of the raw data file
        :return: None
        """
        logger.info("Initiating the data ingestion process")

        try:
            logger.info("Initiating the data ingestion process")
            df = pd.read_csv("Notebooks/Data/stud.csv")
            logger.info("Reading the data from the file as a dataframe")
            
            os.makedirs( os.path.dirname(self.ingestion_path.train_data_path), exist_ok=True )
            
            df.to_csv(self.ingestion_path.train_data_path, index=False, header=True)
            
            logger.info("Train test split of the data is started")
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_path.train_data_path, index=False, header=True)
            
            test_set.to_csv(self.ingestion_path.test_data_path, index=False, header=True)
            
            logger.info("Train test split of the data is completed")
            
            return ( self.ingestion_path.train_data_path, self.ingestion_path.test_data_path )
        except Exception as e:
            raise CustomException( e , sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion("Notebooks/Data/stud.csv")