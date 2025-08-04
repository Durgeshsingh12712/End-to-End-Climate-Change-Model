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