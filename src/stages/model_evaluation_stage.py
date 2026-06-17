import sys

from src.config.configuration import ConfigurationManager
from src.components.model_evaluator import ModelEvaluation
from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException


STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        configuration = ConfigurationManager()

        transformation_config = configuration.get_data_transformation_config()
        model_trainer_config = configuration.get_model_trainer_config()
        model_evaluation_config = configuration.get_model_evaluation_config()

        model_evaluation = ModelEvaluation(
            config=model_evaluation_config,
            data_trans=transformation_config,
            model=model_trainer_config
        )

        model_evaluation.initiate_model_evaluation()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")

        pipeline = ModelEvaluationPipeline()
        pipeline.initiate_model_evaluation()

        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n")

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)