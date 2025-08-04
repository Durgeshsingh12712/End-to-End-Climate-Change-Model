from climateChange.configure import ConfigurationManager
from climateChange.components import (
    DataIngestion
)

class TrainingPipeline:
    def __init__(self):
        pass

    def data_ingestion(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()