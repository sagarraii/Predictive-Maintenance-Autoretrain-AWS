import sys

from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException

from src.stages.data_ingestion import DataIngestionTrainingPipeline
from src.stages.data_validation import DataValidationTrainingPipeline


def run_stage(stage_name: str, pipeline_class):
    try:
        logger.info(f">>>>>>> {stage_name} started <<<<<<")
        obj = pipeline_class()

        if stage_name == "Data Ingestion Stage":
            obj.initiate_data_ingestion()
        elif stage_name == "Data Validation Stage":
            obj.initiate_data_validation()

        logger.info(f">>>>>>> {stage_name} completed <<<<<<\n")
    except Exception as e:
        logger.exception(f"{stage_name} failed.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_stage("Data Ingestion Stage", DataIngestionTrainingPipeline)
    run_stage("Data Validation Stage", DataValidationTrainingPipeline)