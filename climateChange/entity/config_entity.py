from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    climate_data_url: str
    social_data_url: str
    climate_data_path: Path
    social_data_path: Path
    raw_data_path: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    climate_schema: Dict[str, Any]
    social_schema: Dict[str, Any]
    status_file: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str
    max_features: int
    sequence_length: int
    test_size:float
    random_state: int
    preprocessor_path: Path
    transformed_train_path: Path
    transformed_test_path: Path