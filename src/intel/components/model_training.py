import os
import sys
import torch
import joblib
from src.intel.logger import logging
from src.intel.exception import CustomException
from src.intel.constants import *
from src.intel.entity.config_entity import *
from src.intel.entity.artifact_entity import *
from src.intel.utils import *
from src.intel.entity.custom_model import *
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as tt


class ModelTraining():
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact,model_training_config:ModelTrainingConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_training_config = model_training_config

    def get_data_loader(self, train_data):
        try:
            val_size = max(1, int(len(train_data) * 0.2))
            train_size = len(train_data) - val_size

            logging.info("Shuffle and split the training and validation set")
            train_ds , val_ds = random_split(train_data, [train_size, val_size])

            # PyTorch data loader - disable pin_memory when no GPU is available
            use_pin_memory = torch.cuda.is_available()
            logging.info(f"Using pin_memory: {use_pin_memory} (GPU available: {torch.cuda.is_available()})")
            
            training_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers = 0, pin_memory = use_pin_memory)
            valid_dl = DataLoader(val_ds, BATCH_SIZE*2, num_workers = 0 , pin_memory = use_pin_memory)

            logging.info("Exit get_data_loader method of model trainer")

            return training_dl, valid_dl

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def get_model(self, train_data):
        try:
            logging.info("Getting  the pre-trained  model")
            
            num_classes = len(train_data.classes)
            model = ResNet9(3, num_classes)
            return model

        except Exception as e:
            raise CustomException(e, sys) from e

    def load_to_GPU(self, training_dl, valid_dl, model):
        try:
            logging.info("loading model to GPU")
            DEVICE = get_default_device()
            model = to_device(model, DEVICE)

            logging.info("loading model to CPU")
            training_dl = DeviceDataLoader(training_dl, DEVICE)
            valid_dl = DeviceDataLoader(valid_dl, DEVICE)

            logging.info("loading data and model to GPU is done")
            return training_dl, valid_dl, model
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def train_model(self, model, train_dl, valid_dl ):
        try:
            logging.info("Model training started in train_model method")
            logging.info(f"Train data loader batches: {len(train_dl)}")
            logging.info(f"Validation data loader batches: {len(valid_dl)}")
            fitted_model, result = my_fit_method(epochs=EPOCHS, lr = LEARNING_RATE, model=model, train_data_loader=train_dl, val_loader = valid_dl, opt_func = OPTIMIZER,grad_clip=GRAD_CLIP)
            logging.info("Model training is completed")
            return fitted_model, result
        except Exception as e:
            logging.error(f"Error in train_model: {str(e)}")
            raise CustomException(e, sys)


    def initiate_model_training(self):
        try:
            logging.info("Intiate_model_training component started")

            stats = ((0.4301, 0.4574, 0.4537), (0.2482, 0.2467, 0.2807))   
            train_transform = tt.Compose([
                tt.Resize(64),
                tt.RandomCrop(64),
                tt.RandomHorizontalFlip(),
                tt.ToTensor(),
                tt.Normalize(*stats,inplace=True)
            ]) 
            os.makedirs(self.model_training_config.model_training_artifact_dir, exist_ok = True)
            logging.info("Saving transformer oblect for prediction")
            joblib.dump(train_transform, self.model_training_config.transform_object_path)

            logging.info("Applying set of transformations on train dataset")
            dataset_path = self.data_ingestion_artifact.dataset_path
            train_folder_path = os.path.join(dataset_path, "seg_train", "seg_train")
            if os.path.exists(train_folder_path):
                logging.info(f"Using training folder: {train_folder_path}")
                dataset_path = train_folder_path

            train_data = ImageFolder(dataset_path, transform=train_transform)
            if len(train_data.classes) < 2:
                raise CustomException(
                    f"Training dataset must contain at least 2 class subdirectories. "
                    f"Found classes: {train_data.classes}. "
                    f"Dataset path: {dataset_path}", sys
                )

            train_dl, valid_dl = self.get_data_loader(train_data)

            model = self.get_model(train_data)
            torch.cuda.empty_cache()

            logging.info("loading requirements to GPU")
            training_dl, valid_dl, model = self.load_to_GPU(train_dl, valid_dl, model)

            fitted_model, result = self.train_model(model=model,train_dl=training_dl, valid_dl=valid_dl)

            logging.info(f"saving the model at {self.model_training_config.model_path}")
            os.makedirs(self.model_training_config.model_training_artifact_dir, exist_ok=True)
            torch.save(model.state_dict(), self.model_training_config.model_path)
            logging.info(f"Model saved successfully at {self.model_training_config.model_path}")

            model_trainer_artifact = ModelTrainingArtifact(
                model_path = self.model_training_config.model_path,
                result = result,
                transform_object_path = self.model_training_config.transform_object_path

            )

            logging.info(f"Model trainer artifact {model_trainer_artifact}")
            logging.info("model training completed")

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)