import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            return pred
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        gender:str,
        race/ethnicity:int,
        parental level of education,
        lunch: str,
        test preparation course:str,
        reading score:int,
        writing score:int):

        self.gender = gender

        self.race/ethnicity = race/ethnicity

        self.parental level of education = parental level of education

        self.lunch = lunch

        self.test_preparation_course = test preparation course

        self.reading_score = reading score

        self.writing_score = writing score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race/ethnicity],
                "parental level of education": [self.parental level of education],
                "lunch": [self.lunch],
                "test preparation course": [self.test preparation course],
                "reading score": [self.reading score],
                "writing score": [self.writing score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


                     