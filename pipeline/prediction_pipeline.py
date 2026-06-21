import sys
import pandas as pd

from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException
from src.utils.common import load_object

from src.entity.config_entity import (
    DataTransformationConfig,
    ModelTrainerConfig,
)


class PredictionPipeline:
    def __init__(
        self,
        transformation_config: DataTransformationConfig,
        model_config: ModelTrainerConfig,
    ):
        self.transformation_config = transformation_config
        self.model_config = model_config

    def predict(self, features: pd.DataFrame):
        try:
            preprocessor = load_object(
                self.transformation_config.preprocessor_path
            )

            model = load_object(
                self.model_config.best_model_path
            )

            transformed_features = preprocessor.transform(features)

            prediction = model.predict(transformed_features)

            probability = None
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(
                    transformed_features
                )[:, 1]

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

    def get_data_as_dataframe(self):
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