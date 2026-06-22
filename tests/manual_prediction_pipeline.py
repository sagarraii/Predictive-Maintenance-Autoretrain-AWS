from pathlib import Path
import pytest

from pipeline.prediction_pipeline import PredictionPipeline, CustomData
from src.config.configuration import ConfigurationManager

PREPROCESSOR = Path("artifacts/preprocessors/preprocessor.pkl")
MODEL = Path("artifacts/models/best_model.pkl")  # adjust path if yours is different

if not PREPROCESSOR.exists() or not MODEL.exists():
    pytest.skip(
        "Prediction artifacts are not available. Skipping integration test.",
        allow_module_level=True,
    )

def test_prediction_pipeline():
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

    assert prediction is not None
    assert probability is not None

    