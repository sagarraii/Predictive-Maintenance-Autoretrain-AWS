from unittest.mock import patch

from app.app import app


class MockPredictionPipeline:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, df):
        # prediction = Failure, probability = 92%
        return [1], [0.92]


def test_home_page():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200


@patch("app.app.PredictionPipeline", MockPredictionPipeline)
def test_predict_endpoint():
    client = app.test_client()

    payload = {
        "Type": "L",
        "air_temperature": "300",
        "process_temperature": "310",
        "rotational_speed": "1500",
        "torque": "40",
        "tool_wear": "120",
    }

    response = client.post("/predict", data=payload)

    assert response.status_code == 200

    page = response.data.decode("utf-8")

    assert (
        "Machine Failure Predicted" in page
        or "Machine Operating Normally" in page
    )

    assert "92.0000%" in page