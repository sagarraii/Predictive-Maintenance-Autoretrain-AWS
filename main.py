import sys

from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException

from src.stages.data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:

        logger.info(
            f">>>>>>> {STAGE_NAME} started <<<<<<"
        )

        obj = DataIngestionTrainingPipeline()

        obj.initiate_data_ingestion()

        logger.info(
            f">>>>>>> {STAGE_NAME} completed <<<<<<"
        )

except Exception as e:
    raise CustomException(e, sys)