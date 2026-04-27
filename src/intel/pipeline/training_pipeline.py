import os
import sys
from src.intel.logger import logging
from src.intel.exception import CustomException
from src.intel.components.data_ingestion import DataIngestion
from src.intel.components.data_validation import DataValidation
from src.intel.entity.config_entity import *
from src.intel.entity.artifact_entity import *


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
    

    def start_data_ingestion(self)-> DataIngestionArtifact:
        """
        This function is responsible for initiating data ingestion
        """
        logging.info("Entered the start_data_ingestion method of TrainingPipeline class")
        try:
            logging.info("Data Ingestion started")
            data_ingestion_obj = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion_obj.initiate_data_ingestion()
            logging.info("Exited the start_data_ingestion method of TrainingPipeline class")
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)-> DataValidationArtifact:
        """
        This function is responsible for initiating data validation
        """
        logging.info("Entered the start_data_validation method of TrainingPipeline class")
        try:
            logging.info("Data Validation started")
            data_validation_obj = DataValidation(data_ingestion_artifact, self.data_validation_config)
            data_validation_artifact = data_validation_obj.initiate_data_validation()
            logging.info("Exited the start_data_validation method of TrainingPipeline class")
            return data_validation_artifact
        
        except Exception as e:
            raise CustomException(e, sys)


    def run_pipeline(self):
        logging.info("Entered the run_pipeline method of TrainingPipeline class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)

            logging.info("Successfully completed the run_pipeline method of TrainingPipeline class")

        except Exception as e:
            raise e
        