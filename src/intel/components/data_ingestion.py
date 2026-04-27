import os 
import sys
from zipfile import ZipFile 
from src.intel.logger import logging
from src.intel.exception import CustomException
from src.intel.constants import *
from src.intel.entity.artifact_entity import DataIngestionArtifact
from src.intel.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        This function is responsible for data ingestion
        """
        self.data_ingestion_config = data_ingestion_config


    def load_data(self):
        """
        This function is responsible for loading data from the local directory
        """
        logging.info("Data Ingestion started")
        try:
            path = self.data_ingestion_config.ZIP_FILE_PATH
            logging.info(f"Data path loaded: {path}")

            return path
    
        except Exception as e:
            raise CustomException(e, sys)

    
    def unzip(self)-> None:
        """
        This function is responsible for unzipping folder
        """
        logging.info("Entered the unzip method of Data ingestion class")
        try:
            #  Check if zip exists 
            if not os.path.exists(self.data_ingestion_config.ZIP_FILE_PATH):
                logging.info(f"Zip file not found at {self.data_ingestion_config.ZIP_FILE_PATH}")
                logging.info("Assuming data already existed. Skipping unzipping")
                return None
            # Create directories if it doesn't exist
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            logging.info(f" Extracting {self.data_ingestion_config.ZIP_FILE_PATH} to {self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR}")

            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, "r") as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
                logging.info(f"File extracted:{os.listdir(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)}")
            logging.info("Exited the unzip method of Data ingestion class")

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        """
        This function is responsible for initiating data ingestion
        """
        try:
            self.unzip()
            logging.info("Successfully unzip the data from zip file")

            data_ingestion_artifact = DataIngestionArtifact(
                dataset_path = self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR
            )
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e, sys)