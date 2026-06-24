from flask import Flask, render_template, request

from src.utils.s3_utils import S3Sync

from src.config.configuration import ConfigurationManager
from pipeline.prediction_pipeline import (
    PredictionPipeline,
    CustomData,
)

app = Flask(__name__)

configuration = ConfigurationManager()

aws_config = configuration.get_aws_config()

S3Sync.download_folder(
    s3_uri=f"s3://{aws_config.bucket_name}/artifacts/latest",
    local_folder="artifacts",
)

transformation_config = configuration.get_data_transformation_config()
model_trainer_config = configuration.get_model_trainer_config()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = CustomData(
            Type=request.form.get("Type"),
            air_temperature=float(request.form.get("air_temperature")),
            process_temperature=float(request.form.get("process_temperature")),
            rotational_speed=int(request.form.get("rotational_speed")),
            torque=float(request.form.get("torque")),
            tool_wear=int(request.form.get("tool_wear")),
        )

        df = data.get_data_as_dataframe()

        pipeline = PredictionPipeline(
            transformation_config=transformation_config,
            model_config=model_trainer_config,
        )

        prediction, probability = pipeline.predict(df)

        prediction = int(prediction[0])

        if probability is not None:
            probability = round(float(probability[0]), 4)

        result = (
            "⚠️ Machine Failure Predicted"
            if prediction == 1
            else "✅ Machine Operating Normally"
        )

        return render_template(
            "index.html",
            prediction_text=result,
            probability=probability,
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            probability=None,
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )