from src.constant.constants import *
from src.utils.common import read_yaml, create_directories

from src.entity.config_entity import (
    DataIngestionConfig, 
    DataValidationConfig, 
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelTrainerParams,
    RandomSearchParams,
    AWSConfig
    )

class ConfigurationManager:
    def __init__(self,
                 config_file_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH,
                 schema_file_path = SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)

        #------------Data Ingestion------------

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion
        
        data_ingestion_config = DataIngestionConfig(
            dataset_name=config.dataset_name,
            data_download_dir=Path(config.data_download_dir)
        )

        return data_ingestion_config
    
    #----------------------Data Validation---------
    
    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        validation_root = Path(config.root_dir)
        create_directories([validation_root])

        data_validation_config = DataValidationConfig(
            root_dir=validation_root,
            STATUS_FILE=Path(config.STATUS_FILE),
            data_file_path=Path(config.data_file_path),
            all_schema=schema
        )

        return data_validation_config
    
    #------------Data Transformation--------
    
    def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_transformation
        create_directories([Path(config.root_dir), Path(config.preprocessor_path).parent])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_file_path=Path(config.data_file_path),
            train_data_path=Path(config.train_data_path),
            valid_data_path=Path(config.valid_data_path),
            test_data_path=Path(config.test_data_path),

            train_target_path=Path(config.train_target_path),
            valid_target_path=Path(config.valid_target_path),
            test_target_path=Path(config.test_target_path),

            preprocessor_path=Path(config.preprocessor_path)
        )

        return data_transformation_config
    
    #----------Model Trainer-------------------
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:

        config = self.config.model_trainer

        # Convert to Path objects for safe directory generation
        model_root = Path(config.root_dir)
        search_results_dir = Path(config.search_results_dir)

        create_directories([
            model_root,
            search_results_dir
        ])

        model_trainer_config = ModelTrainerConfig(
            root_dir=model_root,
            best_model_path=Path(config.best_model_path),
            best_params_path=Path(config.best_params_path),
            model_report_path=Path(config.model_report_path),
            search_results_dir=search_results_dir
        )

        return model_trainer_config
    
    #------------Model Evaluation-------------

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        config = self.config.model_evaluation

        evaluation_root = Path(config.root_dir)
        create_directories([evaluation_root])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=evaluation_root,
            evaluation_report_path=Path(config.evaluation_report_path),
            classification_report_path=Path(config.classification_report_path),
            confusion_matrix_path=Path(config.confusion_matrix_path),
            model_acceptance_path=Path(config.model_acceptance_path),
            threshold_analysis_path=Path(config.threshold_analysis_path)
        )

        return model_evaluation_config
    
    #----------model Trainer Params-----------
    
    def get_model_trainer_params(self) -> ModelTrainerParams:

        params = self.params

        random_search = RandomSearchParams(
            n_iter=params.random_search.n_iter,
            cv=params.random_search.cv,
            scoring=params.random_search.scoring,
            random_state=params.random_search.random_state,
            n_jobs=params.random_search.n_jobs,
            verbose=params.random_search.verbose
        )

        model_trainer_params = ModelTrainerParams(
            random_search=random_search,

            random_forest=params.random_forest,
            logistic_regression=params.logistic_regression,
            xgboost=params.xgboost,
            catboost=params.catboost
        )

        return model_trainer_params
    
    #----------AWS Configuration-----------
    
    def get_aws_config(self) -> AWSConfig:
        config = self.config.aws
        return AWSConfig(
            bucket_name=config.bucket_name
        )