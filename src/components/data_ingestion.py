import os
import sys

from dotenv import load_dotenv
load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi

from src.entity.config_entity import DataIngestionConfig
from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException


class DataIngestion:
    def __init__(self,
                 config: DataIngestionConfig):

        self.config = config

    def download_data(self):

        try:

            if os.path.exists(self.config.data_download_dir) and os.listdir(self.config.data_download_dir):
                logger.info ("Dataset already exists")
                return 
            
            logger.info(
                "Starting Kaggle dataset download"
            )

            os.makedirs(
                self.config.data_download_dir,
                exist_ok=True
            )

            api = KaggleApi()
            api.authenticate()

            api.dataset_download_files(
                self.config.dataset_name,
                path=self.config.data_download_dir,
                unzip=True
            )

            logger.info(
                "Dataset downloaded successfully"
            )

        except Exception as e:
            raise CustomException(e, sys)