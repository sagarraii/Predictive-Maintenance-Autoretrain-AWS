import sys

from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException


STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_trainer(self):
        configuration = ConfigurationManager()
        transformation_config = (configuration.get_data_transformation_config())
        model_trainer_config = (configuration.get_model_trainer_config())
        model_trainer_params = (configuration.get_model_trainer_params())

        model_trainer = ModelTrainer(
            transformation_config=transformation_config,
            config=model_trainer_config,
            params=model_trainer_params,
        )
        model_trainer.initiate_model_trainer()


if __name__ == "__main__":
    try:

        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")

        pipeline = ModelTrainingPipeline()
        pipeline.initiate_model_trainer()
        
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n")

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)