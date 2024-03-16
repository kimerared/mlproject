import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

from src.exception import CustomException
from src.logs import logging

@dataclass
class DataIngestionConfig:
    train_data_dir: str = 'artifacts/train.csv'
    test_data_dir: str = 'artifacts/test.csv'
    raw_data_dir: str = 'artifacts/raw.csv'

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Initiated")
        try:
            df = pd.read_csv('notebooks/data/stud.csv')  # Dataset
            logging.info("Reading data as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_dir), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_dir, index=False, header=True)

            logging.info("Train/Test Split Initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=99)
            train_set.to_csv(self.ingestion_config.train_data_dir, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_dir, index=False, header=True)
            logging.info("Data Ingestion Completed")

            return self.ingestion_config.train_data_dir, self.ingestion_config.test_data_dir

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    
    result = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(result)
    