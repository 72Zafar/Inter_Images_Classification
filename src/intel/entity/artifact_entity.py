from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    dataset_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool

@dataclass
class ModelTrainingArtifact:
    model_path: str
    result: dict
    transform_object_path: str