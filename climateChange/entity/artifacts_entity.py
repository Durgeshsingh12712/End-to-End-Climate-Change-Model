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
    