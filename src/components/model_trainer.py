import sys
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import (
    DataTransformationConfig,
    ModelTrainerConfig, 
    ModelTrainerParams)

from src.utils.common import (load_numpy_array_data,  save_object, write_yaml_file)


class ModelTrainer:
    def __init__(self, transformation_config: DataTransformationConfig, config: ModelTrainerConfig, params: ModelTrainerParams,):

        self.transformation_config = transformation_config
        self.config = config
        self.params = params


#### set MLflow here

    def load_train_valid_data(self):
        try:
            logger.info("Starting model training: loading training and validation data.")

            X_train = load_numpy_array_data(self.transformation_config.train_data_path)
            X_valid = load_numpy_array_data(self.transformation_config.valid_data_path)
            y_train = load_numpy_array_data(self.transformation_config.train_target_path)
            y_valid = load_numpy_array_data(self.transformation_config.valid_target_path)

            logger.info(
                f"Loaded X_train: {X_train.shape}, "
                f"X_valid: {X_valid.shape}, "
                f"y_train: {y_train.shape}, "
                f"y_valid: {y_valid.shape}"
            )

            return X_train, X_valid, y_train, y_valid

        except Exception as e:
            raise CustomException(e, sys)


    def initialize_models(self):
        try: 
            logger.info("Initializing candidate machine learning models.")
            models = {
                    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42,),
                    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1,),
                    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss",),
                    "CatBoost": CatBoostClassifier(verbose=0, random_state=42,),
                }
            logger.info(f"Initialized {len(models)} models for hyperparameter tuning.")

            return models
        
        except Exception as e:
            raise CustomException (e, sys)


    def get_param_grids(self):
        try:

            logger.info("Loading hyperparameter search spaces for all models.")

            return {
                "Logistic Regression": dict(self.params.logistic_regression),
                "Random Forest": dict(self.params.random_forest),
                "XGBoost": dict(self.params.xgboost),
                "CatBoost": dict(self.params.catboost),
            }
        
        except Exception as e:
            raise CustomException (e, sys)


    def tune_model(self, model, param_grid, X_train, y_train):
        try:
            logger.info(f"Starting RandomizedSearchCV for model: {model}")

            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=self.params.random_search.n_iter,
                cv=self.params.random_search.cv,
                scoring=self.params.random_search.scoring,
                random_state=self.params.random_search.random_state,
                n_jobs=self.params.random_search.n_jobs,
                verbose=self.params.random_search.verbose,
            )
            logger.info("Performing hyperparameter tuning.")

            random_search.fit(X_train, y_train)
            logger.info(
                f"Hyperparameter tuning completed. "
                f"Best CV F1 Score: {random_search.best_score_:.4f}"
            )

            return random_search

        except Exception as e:
            raise CustomException(e, sys)


    def evaluate_model(self, model, X_valid, y_valid):
        try:
            logger.info("Evaluating model on validation dataset.")

            y_pred = model.predict(X_valid)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_valid)[:, 1]
                roc_auc = roc_auc_score(y_valid, y_prob)
            else:
                roc_auc = None

            accuracy = accuracy_score(y_valid, y_pred)
            precision = precision_score(y_valid, y_pred, zero_division=0)
            recall = recall_score(y_valid, y_pred, zero_division=0)
            f1 = f1_score(y_valid, y_pred, zero_division=0)
            roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"

            logger.info(
                f"Validation Metrics - "
                f"Accuracy: {accuracy:.4f}, "
                f"Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, "
                f"F1: {f1:.4f}, "
                f"ROC-AUC: {roc_auc_str}"
            )

            return {
                "Validation Accuracy": round(accuracy, 4),
                "Validation Precision": round(precision, 4),
                "Validation Recall": round(recall, 4),
                "Validation F1": round(f1, 4),
                "Validation ROC-AUC": ( round(roc_auc, 4) if roc_auc is not None else None),
            }

        except Exception as e:
            raise CustomException(e, sys)


    def save_search_results(self, cv_results, model_name):
        try:
            df = pd.DataFrame(cv_results)

            safe_name = (model_name.lower().replace(" ", "_"))
            file_path = (self.config.search_results_dir/ f"{safe_name}_search_results.csv")
            df.to_csv(file_path, index=False)

            logger.info(f"Randomized search results saved to: {file_path}")

        except Exception as e:
            raise CustomException(e, sys)


    def save_artifacts(self, best_model,  best_model_name, best_validation_f1, best_params, report):
        try:
            save_object(file_path=self.config.best_model_path, obj=best_model,)
            logger.info(f"Best model saved at: {self.config.best_model_path}")

            write_yaml_file(file_path=self.config.model_report_path, content=report, replace=True,)
            logger.info(f"Model comparison report saved at: {self.config.model_report_path}")

            write_yaml_file(file_path=self.config.best_params_path,content={
                    "model_name": best_model_name,
                    "validation_f1": float(best_validation_f1),
                    "best_params": best_params,
                },
                replace=True,
            )
            logger.info(f"Best model parameters saved at: {self.config.best_params_path}")

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_model_trainer(self):
        try:
            logger.info("Starting model training pipeline.")
            X_train, X_valid, y_train, y_valid = (self.load_train_valid_data())

            models = self.initialize_models()
            param_grids = self.get_param_grids()
            logger.info("Beginning model selection and hyperparameter tuning.")

            results = []

            best_model = None
            best_model_name = None
            best_validation_f1 = -1.0
            best_params = None

            for model_name, model in models.items():

                logger.info(f"Training {model_name}")

                search = self.tune_model(
                    model=model,
                    param_grid=param_grids[model_name],
                    X_train=X_train,
                    y_train=y_train,
                )

                logger.info(
                    f"Best parameters found for {model_name}: "
                    f"{search.best_params_}"
                )

                self.save_search_results(search.cv_results_, model_name,)
                metrics = self.evaluate_model(search.best_estimator_, X_valid, y_valid,)
                logger.info(
                    f"{model_name} achieved Validation F1 Score: "
                    f"{metrics['Validation F1']:.4f}"
                )

                results.append(
                    {"Model": model_name, "Best CV F1": round(search.best_score_, 4,),
                     **metrics, "Best Params": search.best_params_,
                    }
                )

                if (metrics["Validation F1"] > best_validation_f1):
                    best_validation_f1 = (metrics["Validation F1"])
                    best_model = search.best_estimator_
                    best_model_name = model_name
                    best_params = search.best_params_
                    logger.info(
                        f"{model_name} is the current best model "
                        f"with Validation F1 Score: {best_validation_f1:.4f}"
                    )

            report_df = (pd.DataFrame(results).sort_values(by="Validation F1", ascending=False,).reset_index(drop=True))
            logger.info("Model comparison report generated successfully.")

            self.save_artifacts(
                best_model=best_model,
                best_model_name=best_model_name,
                best_validation_f1=best_validation_f1,
                best_params=best_params,
                report=report_df.to_dict(
                    orient="records"
                ),
            )

            logger.info(
                f"Model training completed successfully. "
                f"Selected best model: {best_model_name} "
                f"(Validation F1: {best_validation_f1:.4f})."
            )

            return self.config.best_model_path

        except Exception as e:
            raise CustomException(e, sys)