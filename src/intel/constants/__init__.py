import os 
import torch

# Data Ingestion related variables
ARIFACTS_DIR: str = "artifacts"
ZIP_FILE_PATH: str = r"D:\My_Folder\New_Mlpos_Coures_projects\Inter_Images_Classification\Data\seg_pred.zip"
ZIP_FILE_NAME: str = "seg_pred.zip"
DATA_INGESTION_ARTIFACTS_DIR = "Data_ingestion"

# Data Validation related variables
SCHEMA_FILE_PATH  = os.path.join("config", "schema.yaml")

# Model Training related variables
MODEL_TRAINING_ARTIFACTS_DIR: str = "Model_training"
MODEL_NAME: str = "model.pt"
BATCH_SIZE: int = 32
EPOCHS: int = 2
LEARNING_RATE: float = 0.001
GRAD_CLIP: float = 0.1
WEIGHT_DECAY: float = 1e-4
IN_CHANNELS: int = 3
OPTIMIZER = torch.optim.RMSprop
NUM_CLASSES: int = 2
TRANSFORM_OBJECT_NAME: str = "transform.pkl"