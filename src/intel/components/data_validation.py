import os
import sys
from src.intel.exception import CustomException
from src.intel.logger import logging
from src.intel.entity.artifact_entity import DataValidationArtifact,DataIngestionArtifact
from src.intel.entity.config_entity import DataValidationConfig
from src.intel.utils import read_yaml_file
from src.intel.constants import *


class DataValidation():
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact):
        self.data_ingestion_artifact = data_ingestion_artifact
        self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

    def count_classes(self,path):
        outcomes = os.listdir(path)
        status = len(outcomes) == len(self._schema_config["classes"])
        return status
    
    def initiate_data_validation(self):
        logging.info("Entered initiate_data_validation method of DataValidation class")
        try:
            logging.info("Data validation started")
            
            validation_error_mes = ""
            
            status = self.count_classes(path = self.data_ingestion_artifact.train_file_path)
            if not status:
                validation_error_mes += "Classes are messing in test data"

            status = self.count_classes(path = self.data_ingestion_artifact.test_file_path)
            if not status:
                validation_error_mes += "Classes are messing in test data"

            validation_status = len(validation_error_mes) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status = validation_status
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)