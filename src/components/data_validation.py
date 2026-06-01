import os
import sys
from datetime import datetime

import pandas as pd

from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    @staticmethod
    def read_data(data_file_path: str) -> pd.DataFrame:
        try:
            logger.info("Reading the dataset")
            df = pd.read_csv(data_file_path)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> tuple[bool, list]:
        try:
            schema_columns = list(self.config.all_schema.keys())
            dataframe_columns = list(dataframe.columns)
            missing_columns = [col for col in schema_columns if col not in dataframe_columns]

            if missing_columns:
                logger.info(f"Missing Columns: {missing_columns}")
                return False, missing_columns

            logger.info("All required columns are present")
            return True, []
        except Exception as e:
            raise CustomException(e, sys)

    def validate_datatypes(self, dataframe: pd.DataFrame) -> tuple[bool, dict]:
        try:
            mismatches = {}
            for column, expected_dtype in self.config.all_schema.items():
                # Guard: skip if column is not present (already flagged by column check)
                if column not in dataframe.columns:
                    continue
                actual_dtype = str(dataframe[column].dtype)
                if actual_dtype != expected_dtype:
                    mismatches[column] = {"expected": expected_dtype, "found": actual_dtype}
                    logger.info(
                        f"Datatype mismatch in '{column}'. "
                        f"Expected: {expected_dtype}, Found: {actual_dtype}"
                    )

            return len(mismatches) == 0, mismatches
        except Exception as e:
            raise CustomException(e, sys)

    def validate_missing_values(self, dataframe: pd.DataFrame) -> tuple[bool, dict]:
        try:
            missing = dataframe.isnull().sum()
            columns_with_nulls = missing[missing > 0].to_dict()

            if columns_with_nulls:
                logger.info(f"Columns with missing values: {columns_with_nulls}")
                return False, columns_with_nulls

            logger.info("No missing values found.")
            return True, {}
        except Exception as e:
            raise CustomException(e, sys)

    def validate_duplicate(self, dataframe: pd.DataFrame) -> tuple[bool, int]:
        try:
            duplicate_count = int(dataframe.duplicated().sum())

            if duplicate_count > 0:
                logger.info(f"Duplicate rows found: {duplicate_count}")
                return False, duplicate_count

            logger.info("No duplicate rows found.")
            return True, 0
        except Exception as e:
            raise CustomException(e, sys)

    def write_validation_report(
        self,
        dataframe: pd.DataFrame,
        overall_status: bool,
        missing_columns: list,
        dtype_mismatches: dict,
        missing_values: dict,
        duplicate_count: int,
    ):
        """Write a rich validation report matching the required format."""
        try:
            rows = dataframe.shape[0]
            cols = dataframe.shape[1]
            numeric_cols = dataframe.select_dtypes(include="number").shape[1]

            lines = [
                "=" * 60,
                "VALIDATION REPORT",
                "=" * 60,
                f"Generated At       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Dataset File       : {self.config.data_file_path}",
                "-" * 60,
                "DATASET OVERVIEW",
                "-" * 60,
                f"Rows (excluding Header) : {rows}",
                f"Columns                 : {cols}",
                f"Numeric Columns         : {numeric_cols}",
                f"Categorical Columns     : {cols - numeric_cols}",
                "-" * 60,
                "VALIDATION CHECKS",
                "-" * 60,
            ]

            # Column check
            col_status = "PASS" if not missing_columns else "FAIL"
            lines.append(f"[{col_status}] Column Presence Check")
            if missing_columns:
                lines.append(f"       Missing Columns: {missing_columns}")

            # Datatype check
            dtype_status = "PASS" if not dtype_mismatches else "FAIL"
            lines.append(f"[{dtype_status}] Datatype Check")
            if dtype_mismatches:
                for col, info in dtype_mismatches.items():
                    lines.append(
                        f"       '{col}': expected={info['expected']}, found={info['found']}"
                    )

            # Missing values check
            mv_status = "PASS" if not missing_values else "FAIL"
            lines.append(f"[{mv_status}] Missing Values Check")
            if missing_values:
                for col, count in missing_values.items():
                    lines.append(f"       '{col}': {count} missing value(s)")

            # Duplicate check
            dup_status = "PASS" if duplicate_count == 0 else "FAIL"
            lines.append(f"[{dup_status}] Duplicate Rows Check")
            if duplicate_count > 0:
                lines.append(f"       Duplicate Rows Found: {duplicate_count}")

            lines += [
                "-" * 60,
                "SUMMARY",
                "-" * 60,
                f"Overall Validation Status : {'PASSED' if overall_status else 'FAILED'}",
                "=" * 60,
            ]

            report_content = "\n".join(lines)

            with open(self.config.STATUS_FILE, "w") as f:
                f.write(report_content)

            logger.info("Validation report written.")
            logger.info(f"\n{report_content}")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self, data_file_path=None):
        try:
            logger.info("Starting data validation")

            # Use config path if not explicitly passed
            path = data_file_path or self.config.data_file_path
            dataframe = self.read_data(path)

            # Run all checks
            col_status, missing_columns = self.validate_number_of_columns(dataframe)
            dtype_status, dtype_mismatches = self.validate_datatypes(dataframe)
            mv_status, missing_values = self.validate_missing_values(dataframe)
            dup_status, duplicate_count = self.validate_duplicate(dataframe)

            overall_status = all([col_status, dtype_status, mv_status, dup_status])

            # Write rich report
            self.write_validation_report(
                dataframe=dataframe,
                overall_status=overall_status,
                missing_columns=missing_columns,
                dtype_mismatches=dtype_mismatches,
                missing_values=missing_values,
                duplicate_count=duplicate_count,
            )

            logger.info(f"Final validation status: {overall_status}")
            return overall_status

        except Exception as e:
            raise CustomException(e, sys)