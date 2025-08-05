from pathlib import Path
from climateChange.constants import *
from climateChange.utils import read_yaml, create_directories
from climateChange.entity import (
    DataIngestionConfig,
    DataValidationConfig,
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([self.config['artifacts_root']])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']
        
        create_directories([config['root_dir']])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config['root_dir'],
            climate_data_url=config['climate_data_url'],
            social_data_url=config['social_data_url'],
            climate_data_path=config['climate_data_path'],
            social_data_path=config['social_data_path'],
            raw_data_path=config['raw_data_path'],
            unzip_dir=config['unzip_dir']
        )
        
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config['data_validation']
        schema = self.schema['schema']

        create_directories([config['root_dir']])

        data_validation_config = DataValidationConfig(
            root_dir=config['root_dir'],
            unzip_data_dir=config['unzip_data_dir'],
            climate_schema=schema['CLIMATE_COLUMNS'],
            social_schema=schema['SOCIAL_COLUMNS'],
            status_file=config['status_file']
        )
        
        return data_validation_config