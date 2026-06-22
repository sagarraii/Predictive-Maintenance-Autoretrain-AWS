import sys
import pandas as pd

from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException
from src.utils.common import load_object

from src.entity.config_entity import (
    DataTransformationConfig,
    ModelTrainerConfig,
)


class PredictPipeline:
    def __init__(
        self,
        transformation_config: DataTransformationConfig,
        model_config: ModelTrainerConfig,
    ):
        self.transformation_config = transformation_config
        self.model_config = model_config

    def predict(self, features: pd.DataFrame):
        """
        Parameters
        ----------
        features : pd.DataFrame
            Raw input dataframe containing:
            - Type
            - Air temperature [K]
            - Process temperature [K]
            - Rotational speed [rpm]
            - Torque [Nm]
            - Tool wear [min]
        """

        try:
            logger.info("Loading preprocessor.")
            preprocessor = load_object(
                self.transformation_config.preprocessor_path
            )

            logger.info("Loading trained model.")
            model = load_object(
                self.model_config.best_model_path
            )

            logger.info("Transforming input features.")
            transformed_features = preprocessor.transform(features)

            logger.info("Generating prediction.")
            prediction = model.predict(transformed_features)

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(
                    transformed_features
                )[:, 1]
            else:
                probability = None

            return prediction, probability

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Type,
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
    ):
        self.Type = Type
        self.air_temperature = air_temperature
        self.process_temperature = process_temperature
        self.rotational_speed = rotational_speed
        self.torque = torque
        self.tool_wear = tool_wear

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Converts user input into a pandas DataFrame with
        the same schema used during training.
        """

        try:
            data = {
                "Type": [self.Type],
                "Air temperature [K]": [self.air_temperature],
                "Process temperature [K]": [self.process_temperature],
                "Rotational speed [rpm]": [self.rotational_speed],
                "Torque [Nm]": [self.torque],
                "Tool wear [min]": [self.tool_wear],
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    print("Predictor module loaded successfully.")

# python -m src.predictor