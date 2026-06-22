from pipeline.prediction_pipeline import PredictionPipeline, CustomData
from src.config.configuration import ConfigurationManager

config = ConfigurationManager()

data = CustomData(
    Type="L",
    air_temperature=300.0,
    process_temperature=310.0,
    rotational_speed=1500,
    torque=40.0,
    tool_wear=50,
)

df = data.get_data_as_dataframe()

pipeline = PredictionPipeline(
    transformation_config=config.get_data_transformation_config(),
    model_config=config.get_model_trainer_config(),
)

prediction, probability = pipeline.predict(df)

print(prediction)
print(probability)


# run the file as module: python -m tests.test_prediction_pipeline 