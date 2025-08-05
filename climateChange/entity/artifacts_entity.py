from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass
class DataIngestionArtifact:
    climate_date_path: Path
    social_data_path: Path
    raw_data_path: Path
    is_ingested: bool
    message: str
    
@dataclass
class DataValidationArtifact:
    validation_status: bool
    climate_data_validation_status: bool
    social_data_validation_status: bool
    missing_columns: Dict[str, list]
    data_drift_status: bool
    validation_report_path: Path
    message: str