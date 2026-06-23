import sys

from src.utils.s3_utils import S3Sync
from src.config.configuration import ConfigurationManager

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
        self.s3_sync = S3Sync()

        config_manager = ConfigurationManager()
        self.aws_config = config_manager.get_aws_config()

    
    def sync_artifacts_to_s3(self):
        latest = f"s3://{self.aws_config.bucket_name}/artifacts/latest"

        self.s3_sync.upload_folder(
            local_folder="artifacts",
            s3_uri=latest,
        )


    def run_pipeline(self):
        try:
            logger.info("========== Training Pipeline Started ==========")

            DataIngestionTrainingPipeline().initiate_data_ingestion()

            DataValidationTrainingPipeline().initiate_data_validation()

            DataTransformationTrainingPipeline().initiate_data_transformation()

            ModelTrainingPipeline().initiate_model_trainer()

            ModelEvaluationPipeline().initiate_model_evaluation()

            self.sync_artifacts_to_s3()

            logger.info("========== Training Pipeline Completed ==========")

        except Exception as e:
            logger.exception("Training pipeline failed.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()