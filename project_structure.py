import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s'
)

list_of_files = [
    #------------ app — flask api + dashboard ----------------

    "app/__init__.py",              # flask app factory — create_app()
    "app/app.py",                   # /predict /health /retrain /metrics /dashboard                               
    "app/templates/index.html",     # sensor input form + risk gauge + machine history table          
    "app/static/style.css",         # minimal dashboard styling                  

    #------------ artifacts — dvc tracked, pushed to s3 --------

    "artifacts/models/",            # best_model.pkl label_encoder.pkl confusion_matrix.png metrics.json                         
    "artifacts/preprocessors/",     # scaler.pkl feature_names.json imputer.pkl

    #------------ config — single source of truth --------------

    "config/schema.yaml",           # ALL paths · hyperparams · AWS settings · thresholds
    "config/config.yaml",           # sensor column names · dtypes · min/max ranges for validation Target Variable

    #------------ data — dvc tracked --------------------------

    "data/raw/",                                              
    "data/processed/",              # train.csv test.csv 

    #------------ ci/cd — github actions + aws codepipeline------------

    ".github/workflows/ci.yml",     # lint + unit tests on every push — blocks deploy on failure               
    ".github/workflows/cd.yml",     # docker build → push ECR → trigger CodePipeline → deploy EC2                  

    #-------------------------- DOCS ------------------------

    "docs/architecture.png",        # system architecture diagram — embed in README
    "docs/api_reference.md" ,       #  all endpoints · request/response examples · error codes           

    # ----------------------- notebooks — eda and experiments ----------------------

    "notebooks/01_eda.ipynb",                            # sensor distributions · failure rate · correlation heatmap · class imbalance                      
    "notebooks/02_feature_engineering.ipynb",            # rolling stats · lag features · RUL calculation · SMOTE    
    "notebooks/03_model_experiments.ipynb",              # XGBoost vs RF vs Isolation Forest comparison table 

    # -------------------------- pipeline — top-level orchestrators ------------------

    "pipeline/training_pipeline.py",                    # calls src/pipelines stages in order — single entry point
    "pipeline/prediction_pipeline.py",                  # loads artifacts → batch scoring on new sensor CSV

    # ----------------------------------- src — main package (all logic lives here) -----------------------------------
    # ----------------------------------- COMPONENTS: one class per pipeline stage — the core logic -----------------------

    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",         # class DataIngestion — pull CSV from S3 → validate schema → save raw            
    "src/components/data_transformation.py",    # class DataTransformation — clean · SMOTE · rolling features · RUL · split
    "src/components/data_validation.py",        # class DataValidation — check dtypes · null % · sensor value ranges vs schema      
    "src/components/model_trainer.py",          # class ModelTrainer — XGBoost + RF supervised · log runs to MLflow · save best     
    "src/components/model_evaluator.py",        # class ModelEvaluator — F1 · PR curve · ROC-AUC · confusion matrix PNG → MLflow    
    "src/components/anomaly_detector.py",       # class AnomalyDetector — Isolation Forest + DBSCAN · compare to supervised     

    # ------------------------------- CONFIG: reads config.yaml and schema.yaml into typed Python dataclasses ----------------------

    "src/config/__init__.py",
    "src/config/configuration.py",             # class ConfigurationManager — read_data_config() · read_model_config() · read_aws_config()

    # ------------------------------------ UTILS: shared helpers — imported by components, no ML logic here -------------------------

    "src/utils/__init__.py",
    "src/utils/common.py",                  # read_yaml() · save_json() · load_json() · save_object() · load_object()                   
    "src/utils/s3_utils.py",                # upload_to_s3() · download_from_s3() · check_s3_key_exists()

    # ------------------------EXCEPTION: custom exception class with file name + line number in message -----------------------------

    "src/exception/__init__.py",
    "src/exception/custom_exception.py",    # class MaintenanceException(Exception) — wraps sys.exc_info() with file+line context

    # ------------------ LOGGING: structured logging — every component imports from here -----------------------

    "src/logger/__init__.py",
    "src/logger/logger.py",                 # setup_logger() — timestamped logs to logs/ dir + stdout · rotating file handler

    # ------------------------ STAGES: stage-level runners — called by top-level pipeline/ orchestrators ---------------------------

    "src/stages/__init__.py",
    "src/stages/data_ingestion.py",             # instantiates DataIngestion · calls initiate() · logs start/end       
    "src/stages/data_validation.py",            # instantiates DataValidation · calls initiate() · raises on failure     
    "src/stages/data_transformation.py",        # instantiates DataTransformation · calls initiate() · saves train/test
    "src/stages/model_trainer.py",              # instantiates ModelTrainer + AnomalyDetector · logs all to MLflow
    "src/stages/model_evaluation.py",           # instantiates ModelEvaluator · saves metrics.json · generates confusion matrix

    # ---------------------------------------ENTITY: typed config dataclasses — what ConfigurationManager returns --------------------------

    "src/entity/__init__.py",
    "src/entity/config_entity.py",              # @dataclass DataIngestionConfig · DataTransformationConfig · ModelTrainerConfig · etc.              
    "src/entity/artifact_entity.py",               

    "src/constants/__init__.py",
    "src/constants/constants.py",                  

    # --------------------------------MONITORING: model health after deployment ---------------------

    "src/monitoring/__init__.py",
    "src/monitoring/drift_detector.py",         # class DriftDetector — compare live sensor dist to training · flag drift · log MLflow
    "src/monitoring/metrics_logger.py",         # class MetricsLogger — Prometheus counters · latency · failure rate per machine ID
                            

    "src/predictor.py",                         # class Predictor — load artifacts once · predict risk · return level + confidence + machine_id


    # ------------------------- tests — pytest --------------------------------

    "tests/__init__.py",
    "tests/test_data_validation.py",                # schema check · null handling · range violations — no model needed                
    "tests/test_data_transformation.py",            # rolling feature shape · scaler output range · no data leakage check
    "tests/test_common_utils.py",                   # read_yaml · save/load_object roundtrip · save_json
    "tests/test_predictor.py",                      # output dict · confidence 0–1 · risk level is valid string
    "tests/test_api.py",                            # /predict 200 · /health 200 · missing fields 400 · bad values 422      


    # ------------------------------------------------ aws codedeploy hooks ----------------------

    "scripts/build_container.sh", 
    "scripts/stop_container.sh",                # docker stop + rm old container
    "scripts/start_container.sh",               # docker run --restart always -p 5000:5000
    "scripts/pull_images.sh",                   # ECR login + docker pull latest
    "scripts/validate_services.sh",             # curl /health × 10 retries — auto-rollback on failure

    # --------------------------- aws lambda — auto-retrain trigger ---------------------------------- 

    "aws/s3_trigger_lambda.py",             # Lambda fn — fires when new sensor CSV lands in S3 → calls CodePipeline
    "aws/iam_policy.json",                  # least-privilege IAM policy for EC2 + CodePipeline + S3 + ECR + Lambda



    "appspec.yml",              # lifecycle hooks — BeforeInstall · AfterInstall · Start · Validate                        
    "buildspec.yml",            # AWS CodeBuild — build Docker → push ECR → write imagedefinitions.json                            

    "dvc.yaml",                 # pipeline DAG — stage_01 → 02 → 03 → 04 → 05 with deps and outs                                 
    "params.yaml",              # DVC-tracked hyperparams — change here, dvc repro, compare in MLflow                           

    "Dockerfile",                         
    "docker-compose.yml",      # app + mlflow UI + prometheus — local dev stack in one command                    
    ".flake8",
    "requirements.txt",        # production deps — scikit-learn · xgboost · mlflow · flask · boto3 · prometheus-client
    "setup.py",                # pip install -e . — makes all src/ sub-packages importable anywhere                         
    "README.md",               # architecture diagram · results table · live URL · quickstart · API reference                               
    "main.py"                  # main entry point for project



    # i created .gitignore directly from git repo and pulled it; by running git pull command in gitbash
    #  NOTE: all the comments has been written by AI (chatgpt)                            
]


for file_path in list_of_files:
    file_path = Path(file_path)

    if str(file_path).endswith("/"):
        file_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory Created: {file_path}")
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            file_path.touch()
            logging.info(f"File Created: {file_path}")

        else:
            logging.info(f"File already exist: {file_path}")


