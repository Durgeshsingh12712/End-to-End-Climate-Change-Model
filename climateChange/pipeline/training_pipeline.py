from climateChange.configure import ConfigurationManager
from climateChange.components import (
    DataIngestion,
    DataValidation
)

class TrainingPipeline:
    def __init__(self):
        pass

    def data_ingestion(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    
    def data_validation(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation_artifact = data_validation.initiate_date_validation()
        