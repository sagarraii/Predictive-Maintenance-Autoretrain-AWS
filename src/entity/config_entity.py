from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_download_dir: Path
    dataset_name: str

@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: Path
    data_file_path: Path #str
    all_schema: dict

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_file_path: Path
    train_data_path: Path
    valid_data_path: Path
    test_data_path: Path

    train_target_path: Path
    valid_target_path: Path
    test_target_path: Path

    preprocessor_path: Path


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    best_model_path: Path
    best_params_path: Path
    model_report_path: Path
    search_results_dir: Path



@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    evaluation_report_path: Path
    classification_report_path: Path
    confusion_matrix_path: Path
    model_acceptance_path: Path
    threshold_analysis_path: Path

@dataclass
class RandomSearchParams:
    n_iter: int
    cv: int
    scoring: str
    random_state: int
    n_jobs: int
    verbose: int

@dataclass
class ModelTrainerParams:
    random_search: RandomSearchParams

    random_forest: dict
    logistic_regression: dict
    xgboost: dict
    catboost: dict


@dataclass(frozen=True)
class AWSConfig:
    bucket_name: str