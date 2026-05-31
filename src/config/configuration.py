import os

from src.constant.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig)
class ConfigurationManager:
    def __init__(self,
                 config_file_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH,
                 schema_file_path = SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        data_ingestion_config = DataIngestionConfig(
            dataset_name=config.dataset_name,
            data_download_dir=config.data_download_dir
        )

        return data_ingestion_config