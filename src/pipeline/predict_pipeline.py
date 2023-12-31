import sys 
from ..logger import logging as logger
from ..exception import CustomException
from ..utils import save_object, load_object
import pandas as pd
import os

class Prediction_Pipeline:
    def __init__(self):
        pass    
    
    def predict(self, data):
        try :
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model = load_object(model_path)
            preprocess = load_object(preprocessor_path)
            data = preprocess.transform(data)
            prediction = model.predict(data)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)
# It is responsible in mapping the input data to the backend model 

class CustomData:
    def __init__(self, gender : str, race_ethnicity : str, parental_level_of_education,
                lunch:str, test_preparation_course : str,  reading_score : int, writing_score : int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_frame(self):
        try:
            CustomData_input_dict = { "gender" : [self.gender],
                                     "race_ethnicity" : [self.race_ethnicity],
                                     "parental_level_of_education" : [self.parental_level_of_education],
                                     "lunch" : [self.lunch],
                                     "test_preparation_course" : [self.test_preparation_course],
                                     "reading_score" : [self.reading_score],
                                     "writing_score" : [self.writing_score]
                                     }
            
            return pd.DataFrame(CustomData_input_dict)
        except Exception as e:
            raise CustomException(e, sys)