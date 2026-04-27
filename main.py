from src.intel.exception import CustomException
from src.intel.logger import logging
import sys
from src.intel.pipeline.training_pipeline import TrainingPipeline


try:
    training_pipeline = TrainingPipeline()
    training_pipeline.run_pipeline()
except Exception as e:
    CustomException(e, sys)
    