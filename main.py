import sys

from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException

from src.stages.data_ingestion_stage import DataIngestionTrainingPipeline
from src.stages.data_validation_stage import DataValidationTrainingPipeline
from src.stages.data_transformation_stage import DataTransformationTrainingPipeline
from src.stages.model_trainer_stage import ModelTrainingPipeline
from src.stages.model_evaluation_stage import ModelEvaluationPipeline


def run_stage(stage_name: str, pipeline_class):
    try:
        logger.info(f">>>>>>> {stage_name} started <<<<<<")
        obj = pipeline_class()

        if stage_name == "Data Ingestion Stage":
            obj.initiate_data_ingestion()

        elif stage_name == "Data Validation Stage":
            obj.initiate_data_validation()

        elif stage_name == "Data Transformation Stage":
            obj.initiate_data_transformation()
        
        elif stage_name == "Model Training Stage":
            obj.initiate_model_trainer()

        elif stage_name == "Model Evaluation Stage":
            obj.initiate_model_evaluation()

        logger.info(f">>>>>>> {stage_name} completed <<<<<<\n")

    except Exception as e:
        logger.exception(f"{stage_name} failed.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_stage("Data Ingestion Stage", DataIngestionTrainingPipeline)
    run_stage("Data Validation Stage", DataValidationTrainingPipeline)
    run_stage("Data Transformation Stage", DataTransformationTrainingPipeline)
    run_stage("Model Training Stage", ModelTrainingPipeline,)
    run_stage("Model Evaluation Stage", ModelEvaluationPipeline)