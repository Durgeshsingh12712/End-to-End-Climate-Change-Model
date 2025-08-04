import sys
from climateChange.loggers import logger

def error_message_detail(error, error_detail: sys):
    """Custom Error Message with Detailed Information"""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CCException(Exception):
    """Custom Exception Class for Climate Change Modeling Project"""

    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_details)

    def __str__(self):
        return self.error_message

class DataIngestionException(CCException):
    """Exception raised during Data Ingestion"""
    pass

class DataValidationException(CCException):
    """Exception raised during Data Validation"""
    pass

class DataTransformationException(CCException):
    """Exception raised during Data Transformation"""
    pass

class ModelTrainerException(CCException):
    """Exception raised during Model Trainer"""
    pass

class ModelEvaluationException(CCException):
    """Exception raised during Model Evaluation"""
    pass