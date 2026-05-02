import os
import sys
from src.intel.exception import CustomException
from src.intel.logger import logging
from src.intel.entity.artifact_entity import DataValidationArtifact,DataIngestionArtifact
from src.intel.entity.config_entity import DataValidationConfig
from src.intel.utils import read_yaml_file
from src.intel.constants import *


class DataValidation():
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self._schema_config = read_yaml_file(file_path=self.data_validation_config.schema_file_path)

    def count_classes(self, path):
        outcomes = os.listdir(path)
        status = len(outcomes) == len(self._schema_config["classes"])
        return status
    
    def initiate_data_validation(self):
        logging.info("Entered initiate_data_validation method of DataValidation class")
        try:
            logging.info("Data validation started")
            
            validation_error_mes = ""
            
            # Build paths for train and test directories
            train_path = os.path.join(self.data_ingestion_artifact.dataset_path, "seg_train", "seg_train")
            test_path = os.path.join(self.data_ingestion_artifact.dataset_path, "seg_test", "seg_test")
            
            logging.info(f"Train path: {train_path}")
            logging.info(f"Test path: {test_path}")
            logging.info(f"Expected classes: {self._schema_config['classes']}")
            
            if os.path.exists(train_path):
                train_classes = os.listdir(train_path)
                logging.info(f"Train classes found: {train_classes}")
                status = len(train_classes) == len(self._schema_config["classes"])
                if not status:
                    validation_error_mes += "Classes are missing in train data"
            else:
                logging.warning(f"Train path does not exist: {train_path}")
                validation_error_mes += "Train path not found"
            
            if os.path.exists(test_path):
                test_classes = os.listdir(test_path)
                logging.info(f"Test classes found: {test_classes}")
                status = len(test_classes) == len(self._schema_config["classes"])
                if not status:
                    validation_error_mes += "Classes are missing in test data"
            else:
                logging.warning(f"Test path does not exist: {test_path}")
                validation_error_mes += "Test path not found"

            validation_status = len(validation_error_mes) == 0
            logging.info(f"Validation status: {validation_status}")
            logging.info(f"Validation errors: {validation_error_mes}")

            data_validation_artifact = DataValidationArtifact(
                validation_status = validation_status
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            logging.error(f"Error in data validation: {str(e)}")
            raise CustomException(e, sys)