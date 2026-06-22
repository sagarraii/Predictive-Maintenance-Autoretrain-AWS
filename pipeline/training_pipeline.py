import sys

from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException

from src.stages.data_ingestion_stage import DataIngestionTrainingPipeline
from src.stages.data_validation_stage import DataValidationTrainingPipeline
from src.stages.data_transformation_stage import (
    DataTransformationTrainingPipeline,
)
from src.stages.model_trainer_stage import ModelTrainingPipeline
from src.stages.model_evaluation_stage import ModelEvaluationPipeline


class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logger.info("========== Training Pipeline Started ==========")

            DataIngestionTrainingPipeline().initiate_data_ingestion()

            DataValidationTrainingPipeline().initiate_data_validation()

            DataTransformationTrainingPipeline().initiate_data_transformation()

            ModelTrainingPipeline().initiate_model_trainer()

            ModelEvaluationPipeline().initiate_model_evaluation()

            logger.info("========== Training Pipeline Completed ==========")

        except Exception as e:
            logger.exception("Training pipeline failed.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()