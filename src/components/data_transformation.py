import sys
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from imblearn.over_sampling import SMOTE

from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import DataTransformationConfig

from src.utils.common import (
    save_numpy_array_data,
    save_object
)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    @staticmethod
    def read_data(data_file_path: Path) -> pd.DataFrame:
        try:
            logger.info("Starting data transformation")
            logger.info("Reading the dataset")

            df = pd.read_csv(data_file_path)

            logger.info(f"Dataset loaded with shape: {df.shape}")
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def drop_useless_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            columns_to_drop = [
                "UDI",
                "Product ID", 
                "Failure Type"  #removing this to prevent from info leakage of target column 
            ]

            df=df.drop(columns=columns_to_drop, errors="ignore")

            logger.info(f"Dropped columns: {columns_to_drop}")
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def split_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        try:
            X = df.drop(columns=["Target"])
            y = df["Target"]
            return X, y
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def train_test_split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        try:
            X, y = self.split_features_target(df)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

            X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

            logger.info(
                f"Feature shapes (X)- "
                f"Train: {X_train.shape}, "
                f"Validation: {X_valid.shape}, "
                f"Test: {X_test.shape}"
            )

            logger.info(
                f"Target shapes (y) - "
                f"Train: {y_train.shape}, "
                f"Validation: {y_valid.shape}, "
                f"Test: {y_test.shape}"
            )
            return X_train, X_valid, X_test, y_train, y_valid, y_test
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def identify_feature_types(self, X_train: pd.DataFrame) -> tuple[list[str], list[str]]:
        try:
            num_features = list(X_train.select_dtypes(include=np.number).columns)

            cat_features = list(X_train.select_dtypes(include=["object", "category"]).columns)

            logger.info(f"Numerical Features: {list(num_features)}")

            logger.info(f"Categorical Features: {list(cat_features)}")
            return num_features, cat_features
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_preprocessor(self, numerical_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
        '''note in my dataset there was only one Categorical col,
        if in your data there are more the 2 Categorical cols you need to modify this function as per you requirements'''

        try:
            ordinal_mapping = [["L", "M", "H"]]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("ordinal", OrdinalEncoder(categories=ordinal_mapping), categorical_features),
                    ("numeric", StandardScaler(),numerical_features)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def transform_data(self, preprocessor: ColumnTransformer, X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            X_train_processed = preprocessor.fit_transform(X_train)

            X_valid_processed = preprocessor.transform(X_valid)

            X_test_processed = preprocessor.transform(X_test)

            logger.info("Data preprocessing completed")
            return X_train_processed, X_valid_processed, X_test_processed
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def apply_smote(self, X_train_processed: np.ndarray, y_train: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        try:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = (smote.fit_resample(X_train_processed,y_train))
           
            logger.info(
                f"Training data after SMOTE - "
                f"X: {X_train_resampled.shape}, "
                f"y: {y_train_resampled.shape}"
            )

            return X_train_resampled, y_train_resampled
        
        except Exception as e:
            raise CustomException(e, sys)

    def save_transformed_data(self, 
                              X_train_resampled: np.ndarray, 
                              y_train_resampled: np.ndarray, 
                              X_valid_processed: np.ndarray, 
                              y_valid: np.ndarray, 
                              X_test_processed: np.ndarray, 
                              y_test: np.ndarray, 
                              preprocessor: ColumnTransformer,) -> None:
        try:
            save_object(file_path=self.config.preprocessor_path,obj=preprocessor)

            save_numpy_array_data(self.config.train_data_path, X_train_resampled)
            save_numpy_array_data(self.config.valid_data_path, X_valid_processed)
            save_numpy_array_data(self.config.test_data_path, X_test_processed)
            save_numpy_array_data(self.config.train_target_path,np.asarray(y_train_resampled))
            save_numpy_array_data(self.config.valid_target_path, np.asarray(y_valid))
            save_numpy_array_data(self.config.test_target_path, np.asarray(y_test))

            logger.info("Processed numpy arrays saved successfully")
            
        except Exception as e:
                raise CustomException(e, sys)
    
    def initiate_data_transformation(self) -> tuple:
        try:
            df = self.read_data(self.config.data_file_path)
            df = self.drop_useless_cols(df)
            (X_train, X_valid, X_test, y_train, y_valid, y_test,) = self.train_test_split_data(df)
            numerical_features, categorical_features = self.identify_feature_types(X_train)

            # Create preprocessing pipeline
            preprocessor = self.create_preprocessor(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
            )

            # Apply preprocessing
            (X_train_processed, X_valid_processed, X_test_processed,) = self.transform_data(preprocessor=preprocessor, 
                                                                                            X_train=X_train, 
                                                                                            X_valid=X_valid, 
                                                                                            X_test=X_test,)

            # Apply SMOTE only on training data
            (X_train_resampled, y_train_resampled,) = self.apply_smote(X_train_processed=X_train_processed, y_train=y_train,)

            # Save preprocessor and processed datasets
            self.save_transformed_data(
                X_train_resampled=X_train_resampled,
                y_train_resampled=y_train_resampled,
                X_valid_processed=X_valid_processed,
                y_valid=y_valid,
                X_test_processed=X_test_processed,
                y_test=y_test,
                preprocessor=preprocessor,
            )

            logger.info("Data transformation completed successfully.")
            return (
                self.config.train_data_path,
                self.config.valid_data_path,
                self.config.test_data_path,
                self.config.train_target_path,
                self.config.valid_target_path,
                self.config.test_target_path,
                self.config.preprocessor_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        


        