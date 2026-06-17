import sys

from src.config.configuration import ConfigurationManager
from src.components.data_validation import DataValidation
from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException

STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.initiate_data_validation(
        data_file_path = data_validation_config.data_file_path)

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        data_validation_pipeline = DataValidationTrainingPipeline()
        data_validation_pipeline.initiate_data_validation()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\n")
    except Exception as e:
        logger.info(e)
        raise CustomException(e, sys)