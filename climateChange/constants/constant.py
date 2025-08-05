import os
from pathlib import Path

# Base paths
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")
SCHEMA_FILE_PATH = Path("schema.yaml")


# Artifact paths
DATA_INGESTION_ARTIFACTS = Path("artifacts/data_ingestion")
DATA_VALIDATION_ARTIFACTS = Path("artifacts/data_validation")
DATA_TRANSFORMATION_ARTIFACTS = Path("artifacts/data_transformation")
MODEL_TRAINER_ARTIFACTS = Path("artifacts/model_trainer")
MODEL_EVALUATION_ARTIFACTS = Path("artifacts/model_evaluation")

# Schema columns
CLIMATE_SCHEMA_COLUMNS = [
    "date", "temperature_anomaly", "precipitation_anomaly", 
    "co2_concentration", "sea_level_change", "solar_radiation"
]

SOCIAL_SCHEMA_COLUMNS = [
    "date", "text", "sentiment", "likesCount", "commentsCount", "profileName"
]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "temperature_anomaly"

# File extensions
ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.json']

# API endpoints
NASA_API_BASE = "https://climate.nasa.gov/api"
NOAA_API_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"