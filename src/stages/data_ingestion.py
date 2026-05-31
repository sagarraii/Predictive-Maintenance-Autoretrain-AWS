import sys

from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:

    def __init__(self):
        pass

    def initiate_data_ingestion(self):

        config = ConfigurationManager()

        data_ingestion_config = (
            config.get_data_ingestion_config()
        )

        data_ingestion = DataIngestion(
            config=data_ingestion_config
        )

        data_ingestion.download_data()


if __name__ == "__main__":

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