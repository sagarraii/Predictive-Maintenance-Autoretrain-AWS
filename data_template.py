import pandas as pd
import yaml
import sys

from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException
df = pd.read_csv("data/raw/predictive_maintenance.csv")

schema = {
    "COLUMNS":{
        col: str(dtype)
        for col, dtype in df.dtypes.items()
    },

    "TARGET_COLUMNS":{
        "target_1": "Target",
        "target_2": "Failure Type"
    }
}

try: 
    with open ("config/schema.yaml", "w") as file:
        yaml.dump(schema, file)

    logger.info("schema.yaml generated successfully")
except Exception as e:
    raise CustomException(e, sys)