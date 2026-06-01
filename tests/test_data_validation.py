import os
import sys
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig


# ──────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────

SCHEMA = {
    "UDI": "int64",
    "Product ID": "object",
    "Type": "object",
    "Air temperature [K]": "float64",
    "Process temperature [K]": "float64",
    "Rotational speed [rpm]": "int64",
    "Torque [Nm]": "float64",
    "Tool wear [min]": "int64",
    "Target": "int64",
    "Failure Type": "object",
}


def make_valid_df(n=10) -> pd.DataFrame:
    return pd.DataFrame({
        "UDI": range(1, n + 1),
        "Product ID": [f"M{i}" for i in range(n)],
        "Type": ["M"] * n,
        "Air temperature [K]": [300.0] * n,
        "Process temperature [K]": [310.0] * n,
        "Rotational speed [rpm]": [1500] * n,
        "Torque [Nm]": [40.0] * n,
        "Tool wear [min]": [10] * n,
        "Target": [0] * n,
        "Failure Type": ["No Failure"] * n,
    })


@pytest.fixture
def tmp_status_file(tmp_path):
    return str(tmp_path / "status.txt")


@pytest.fixture
def validator(tmp_status_file):
    config = DataValidationConfig(
        root_dir=Path(tmp_status_file).parent,
        STATUS_FILE=tmp_status_file,
        data_file_path=Path("dummy.csv"),
        all_schema=SCHEMA,
    )
    return DataValidation(config=config)


# ──────────────────────────────────────────────
# COLUMN VALIDATION
# ──────────────────────────────────────────────

class TestValidateNumberOfColumns:

    def test_all_columns_present(self, validator):
        df = make_valid_df()
        status, missing = validator.validate_number_of_columns(df)
        assert status is True
        assert missing == []

    def test_missing_one_column(self, validator):
        df = make_valid_df().drop(columns=["Target"])
        status, missing = validator.validate_number_of_columns(df)
        assert status is False
        assert "Target" in missing

    def test_missing_multiple_columns(self, validator):
        df = make_valid_df().drop(columns=["Target", "Failure Type", "UDI"])
        status, missing = validator.validate_number_of_columns(df)
        assert status is False
        assert len(missing) == 3

    def test_extra_columns_still_pass(self, validator):
        df = make_valid_df()
        df["extra_col"] = 99
        status, missing = validator.validate_number_of_columns(df)
        assert status is True


# ──────────────────────────────────────────────
# DATATYPE VALIDATION
# ──────────────────────────────────────────────

class TestValidateDatatypes:

    def test_correct_dtypes(self, validator):
        df = make_valid_df()
        status, mismatches = validator.validate_datatypes(df)
        assert status is True
        assert mismatches == {}

    def test_wrong_dtype_in_one_column(self, validator):
        df = make_valid_df()
        df["Target"] = df["Target"].astype(str)   # int64 -> object
        status, mismatches = validator.validate_datatypes(df)
        assert status is False
        assert "Target" in mismatches

    def test_skips_missing_column_without_crash(self, validator):
        """Should not raise KeyError when a column is absent — handled by guard."""
        df = make_valid_df().drop(columns=["Target"])
        status, mismatches = validator.validate_datatypes(df)
        # Target is missing → skipped; no other mismatches
        assert "Target" not in mismatches


# ──────────────────────────────────────────────
# MISSING VALUES VALIDATION
# ──────────────────────────────────────────────

class TestValidateMissingValues:

    def test_no_missing_values(self, validator):
        df = make_valid_df()
        status, nulls = validator.validate_missing_values(df)
        assert status is True
        assert nulls == {}

    def test_missing_values_detected(self, validator):
        df = make_valid_df()
        df.loc[0, "Torque [Nm]"] = None
        df.loc[1, "Torque [Nm]"] = None
        status, nulls = validator.validate_missing_values(df)
        assert status is False
        assert "Torque [Nm]" in nulls
        assert nulls["Torque [Nm]"] == 2


# ──────────────────────────────────────────────
# DUPLICATE VALIDATION
# ──────────────────────────────────────────────

class TestValidateDuplicate:

    def test_no_duplicates(self, validator):
        df = make_valid_df()
        status, count = validator.validate_duplicate(df)
        assert status is True
        assert count == 0

    def test_duplicates_detected(self, validator):
        df = make_valid_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        status, count = validator.validate_duplicate(df)
        assert status is False
        assert count == 1


# ──────────────────────────────────────────────
# REPORT WRITING
# ──────────────────────────────────────────────

class TestWriteValidationReport:

    def test_report_file_created(self, validator, tmp_status_file):
        df = make_valid_df()
        validator.write_validation_report(
            dataframe=df,
            overall_status=True,
            missing_columns=[],
            dtype_mismatches={},
            missing_values={},
            duplicate_count=0,
        )
        assert os.path.exists(tmp_status_file)

    def test_report_contains_key_fields(self, validator, tmp_status_file):
        df = make_valid_df()
        validator.write_validation_report(
            dataframe=df,
            overall_status=True,
            missing_columns=[],
            dtype_mismatches={},
            missing_values={},
            duplicate_count=0,
        )
        content = open(tmp_status_file).read()
        assert "VALIDATION REPORT" in content
        assert "Rows (excluding Header)" in content
        assert "PASSED" in content

    def test_report_shows_failed_on_issues(self, validator, tmp_status_file):
        df = make_valid_df()
        validator.write_validation_report(
            dataframe=df,
            overall_status=False,
            missing_columns=["Target"],
            dtype_mismatches={},
            missing_values={},
            duplicate_count=3,
        )
        content = open(tmp_status_file).read()
        assert "FAILED" in content
        assert "Target" in content
        assert "3" in content


# ──────────────────────────────────────────────
# FULL PIPELINE (integration)
# ──────────────────────────────────────────────

class TestInitiateDataValidation:

    def test_valid_csv_passes(self, validator, tmp_path):
        csv_path = tmp_path / "test.csv"
        make_valid_df().to_csv(csv_path, index=False)
        result = validator.initiate_data_validation(data_file_path=str(csv_path))
        assert result is True

    def test_csv_with_missing_column_fails(self, validator, tmp_path):
        csv_path = tmp_path / "test.csv"
        make_valid_df().drop(columns=["Target"]).to_csv(csv_path, index=False)
        result = validator.initiate_data_validation(data_file_path=str(csv_path))
        assert result is False

    def test_csv_with_nulls_fails(self, validator, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = make_valid_df()
        df.loc[0, "Torque [Nm]"] = None
        df.to_csv(csv_path, index=False)
        result = validator.initiate_data_validation(data_file_path=str(csv_path))
        assert result is False

    def test_csv_with_duplicates_fails(self, validator, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = make_valid_df()
        pd.concat([df, df.iloc[[0]]]).to_csv(csv_path, index=False)
        result = validator.initiate_data_validation(data_file_path=str(csv_path))
        assert result is False

    def test_nonexistent_file_raises(self, validator):
        with pytest.raises(Exception):
            validator.initiate_data_validation(data_file_path="nonexistent.csv")