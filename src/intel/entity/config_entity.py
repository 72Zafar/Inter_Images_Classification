from datetime import datetime
from src.intel.constants import *
from dataclasses import dataclass


TIMESTEMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# @dataclass
# class DataIngestionConfig:
#     data_ingestion_artifact_dir = os.path.join(from_root(), ARIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
#     data_path = os.path.join(data_ingestion_artifact_dir, TRAIN_FOLDER_LOCATION, TEST_FOLDER_LOCATION, PRED_FOLDER_LOCATION)
#     train_path = os.path.join(data_path, TRAIN_FOLDER_NAME)
#     test_path = os.path.join(data_path, TEST_FOLDER_NAME)
#     pred_path = os.path.join(data_path, PRED_FOLDER_NAME)

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.data_ingestion_dir = os.path.join(os.getcwd(),ARIFACTS_DIR,DATA_INGESTION_ARTIFACTS_DIR)

        # Create directories first - befor any codr runs
        os.makedirs(self.data_ingestion_dir,exist_ok=True)
        self.ZIP_FILE_PATH = ZIP_FILE_PATH
        self.DATA_INGESTION_ARTIFACTS_DIR = self.data_ingestion_dir


