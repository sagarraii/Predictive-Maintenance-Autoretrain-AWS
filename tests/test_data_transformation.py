from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig


@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    
    n = 100

    target = np.array([0] * 90 + [1] * 10)
    np.random.shuffle(target)

    return pd.DataFrame({
        "UDI": range(n),
        "Product ID": [f"P{i}" for i in range(n)],
        "Type": np.random.choice(["L", "M", "H"], n),
        "Air temperature [K]": np.random.normal(300, 2, n),
        "Process temperature [K]": np.random.normal(310, 2, n),
        "Rotational speed [rpm]": np.random.randint(1000, 2000, n),
        "Torque [Nm]": np.random.uniform(20, 80, n),
        "Tool wear [min]": np.random.randint(0, 250, n),
        "Failure Type": np.random.choice(
            ["No Failure", "Heat Dissipation Failure"], n
        ),
        "Target": target,
    })


@pytest.fixture
def transformation_config(tmp_path):
    return DataTransformationConfig(
        root_dir=tmp_path,
        data_file_path=tmp_path / "dummy.csv",
        train_data_path=tmp_path / "X_train.npy",
        valid_data_path=tmp_path / "X_valid.npy",
        test_data_path=tmp_path / "X_test.npy",
        train_target_path=tmp_path / "y_train.npy",
        valid_target_path=tmp_path / "y_valid.npy",
        test_target_path=tmp_path / "y_test.npy",
        preprocessor_path=tmp_path / "preprocessor.pkl",
    )


@pytest.fixture
def transformer(transformation_config):
    return DataTransformation(config=transformation_config)


def test_drop_useless_cols(transformer, sample_dataframe):
    df = transformer.drop_useless_cols(sample_dataframe)

    assert "UDI" not in df.columns
    assert "Product ID" not in df.columns
    assert "Failure Type" not in df.columns


def test_split_features_target(transformer, sample_dataframe):
    df = transformer.drop_useless_cols(sample_dataframe)

    X, y = transformer.split_features_target(df)

    assert "Target" not in X.columns
    assert len(X) == len(y)


def test_train_validation_test_split(transformer, sample_dataframe):
    df = transformer.drop_useless_cols(sample_dataframe)

    X_train, X_valid, X_test, y_train, y_valid, y_test = (
        transformer.train_test_split_data(df)
    )

    total = len(df)

    assert (
        len(X_train)
        + len(X_valid)
        + len(X_test)
        == total
    )

    assert (
        len(y_train)
        + len(y_valid)
        + len(y_test)
        == total
    )


def test_identify_feature_types(transformer, sample_dataframe):
    df = transformer.drop_useless_cols(sample_dataframe)

    X, _ = transformer.split_features_target(df)

    num_features, cat_features = transformer.identify_feature_types(X)

    assert "Type" in cat_features
    assert "Torque [Nm]" in num_features


def test_create_preprocessor(transformer):
    preprocessor = transformer.create_preprocessor(
        numerical_features=[
            "Air temperature [K]",
            "Torque [Nm]",
        ],
        categorical_features=["Type"],
    )

    assert isinstance(preprocessor, ColumnTransformer)


def test_transform_data(transformer, sample_dataframe):
    df = transformer.drop_useless_cols(sample_dataframe)

    X_train, X_valid, X_test, *_ = (
        transformer.train_test_split_data(df)
    )

    num_features, cat_features = (
        transformer.identify_feature_types(X_train)
    )

    preprocessor = transformer.create_preprocessor(
        num_features,
        cat_features,
    )

    X_train_p, X_valid_p, X_test_p = transformer.transform_data(
        preprocessor,
        X_train,
        X_valid,
        X_test,
    )

    assert X_train_p.shape[1] == X_valid_p.shape[1]
    assert X_train_p.shape[1] == X_test_p.shape[1]


def test_apply_smote(transformer, sample_dataframe):
    df = transformer.drop_useless_cols(sample_dataframe)

    X_train, _, _, y_train, _, _ = (
        transformer.train_test_split_data(df)
    )

    num_features, cat_features = (
        transformer.identify_feature_types(X_train)
    )

    preprocessor = transformer.create_preprocessor(
        num_features,
        cat_features,
    )

    X_train_p, _, _ = transformer.transform_data(
        preprocessor,
        X_train,
        X_train,
        X_train,
    )

    X_resampled, y_resampled = transformer.apply_smote(
        X_train_p,
        y_train,
    )

    unique, counts = np.unique(y_resampled, return_counts=True)

    assert len(unique) == 2
    assert counts[0] == counts[1]



'''
Run all tests: pytest
run only this file: pytest tests/test_data_transformation.py -v
To run a single test: pytest tests/test_data_transformation.py -k apply_smote -v
'''