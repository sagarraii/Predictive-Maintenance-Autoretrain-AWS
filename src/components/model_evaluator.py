import sys
import numpy as np
from sklearn.base import BaseEstimator
from pathlib import Path
import matplotlib.pyplot as plt
from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import (
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig)

from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             roc_auc_score, 
                             matthews_corrcoef, 
                             confusion_matrix, 
                             classification_report)

from src.utils.common import (load_numpy_array_data, 
                              load_object,
                              write_yaml_file)


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, data_trans: DataTransformationConfig, model: ModelTrainerConfig) -> None:
        self.config = config
        self.data_trans = data_trans
        self.model = model
    

    def read_data(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            logger.info("Loading test dataset.")

            X_test = load_numpy_array_data(self.data_trans.test_data_path)
            y_test = load_numpy_array_data(self.data_trans.test_target_path)

            logger.info(f"Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")
            return X_test, y_test
        
        except Exception as e:
            raise CustomException(e, sys)
    

    def load_model(self) -> BaseEstimator:
        try:
            logger.info(f"Loading trained model from {self.model.best_model_path}")

            best_model = load_object(self.model.best_model_path)

            logger.info("Best model loaded successfully.")
            return best_model
        
        except Exception as e:
            raise CustomException(e, sys)
    

    def evaluate_model(self, model, X_test, y_test) -> tuple[dict, np.ndarray, np.ndarray]:
        try:
            logger.info("Evaluating model on Test dataset.")

            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            else:
                roc_auc = None

            metrics = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
                "matthews_corrcoef": round(matthews_corrcoef(y_test, y_pred), 4),
            }

            roc_auc_str = (
                f"{metrics['roc_auc']:.4f}"
                if metrics["roc_auc"] is not None
                else "N/A"
            )

            logger.info(
                f"Test Metrics - "
                f"Accuracy: {metrics['accuracy']:.4f}, "
                f"Balanced Accuracy Score: {metrics['balanced_accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}, "
                f"ROC-AUC: {roc_auc_str}, "
                f"Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}"
            )

            probabilities = y_prob if roc_auc is not None else None

            return metrics, y_pred, probabilities          

        except Exception as e:
            raise CustomException(e, sys)  
    

    def generate_classification_report(self, y_test, y_pred) -> dict:
        try:
            
            cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0,)

            logger.info("Classification report generated successfully.")
            return cr
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def save_report(self, report, metrics) -> None:
        try:
            write_yaml_file(file_path=self.config.classification_report_path, content=report, replace=True,)

            logger.info(
                f"Classification report saved to "
                f"{self.config.classification_report_path}"
            )

            write_yaml_file(file_path=self.config.evaluation_report_path, content=metrics, replace=True,)

            logger.info(
                f"Evaluation report saved to "
                f"{self.config.evaluation_report_path}"
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        

    def save_model_acceptance(self, metrics) -> None:
        try:
            threshold = 0.65
            accepted = metrics["f1"] >= threshold

            report = {
                "accepted": accepted,
                "criterion": f"f1 >= {threshold}",
                "actual_f1": metrics["f1"],
            }

            write_yaml_file(file_path=self.config.model_acceptance_path, content=report, replace=True,)

            logger.info(
                f"Model acceptance report saved to "
                f"{self.config.model_acceptance_path}"
            )

        except Exception as e:
            raise CustomException(e, sys)
    

    def save_threshold_analysis(self, y_test, y_prob) -> None:
        try:
            if y_prob is None:
                logger.info(
                    "Model does not support predict_proba. "
                    "Skipping threshold analysis."
                )
                return

            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            analysis = []

            for t in thresholds:
                preds = (y_prob >= t).astype(int)

                analysis.append({
                    "threshold": t,
                    "precision": round(
                        precision_score(y_test, preds, zero_division=0), 4
                    ),
                    "recall": round(
                        recall_score(y_test, preds, zero_division=0), 4
                    ),
                    "f1": round(
                        f1_score(y_test, preds, zero_division=0), 4
                    ),
                })

            write_yaml_file(file_path=self.config.threshold_analysis_path, content=analysis, replace=True,)

            logger.info(
                f"Threshold analysis saved to "
                f"{self.config.threshold_analysis_path}"
            )
            
        except Exception as e:
            raise CustomException(e, sys)


    def save_confusion_matrix(self, y_test, y_pred) -> None:
        try:
            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(5, 5))
            plt.imshow(cm, interpolation="nearest", aspect="equal")
            plt.title("Confusion Matrix")
            plt.colorbar()

            plt.xticks([0, 1], ["No Failure", "Failure"])
            plt.yticks([0, 1], ["No Failure", "Failure"])

            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center")

            plt.tight_layout()
            self.config.confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True,)
            plt.savefig(self.config.confusion_matrix_path, bbox_inches="tight",)
            plt.close()
            logger.info(
                f"Confusion matrix saved to "
                f"{self.config.confusion_matrix_path}"
            )

        except Exception as e:
            raise CustomException(e, sys)
    

    def initiate_model_evaluation(self) -> Path:
        try: 
            logger.info("Starting model evaluation pipeline.")

            X_test, y_test = self.read_data()
            model = self.load_model()

            metrics, y_pred, y_prob = self.evaluate_model(model=model, X_test=X_test, y_test=y_test,)
            report = self.generate_classification_report(y_test=y_test, y_pred=y_pred,)
            self.save_report(report=report, metrics=metrics,)
            self.save_model_acceptance(metrics)
            self.save_threshold_analysis(y_test=y_test,y_prob=y_prob,)
            self.save_confusion_matrix(y_test=y_test, y_pred=y_pred,)

            logger.info("Model evaluation completed successfully.")
            return self.config.evaluation_report_path
        
        except Exception as e:
            raise CustomException(e, sys)
