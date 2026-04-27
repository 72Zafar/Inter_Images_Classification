from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    dataset_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool