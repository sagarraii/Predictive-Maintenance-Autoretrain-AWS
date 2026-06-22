import numpy as np
from pathlib import Path

from src.utils.common import (
    read_yaml,
    write_yaml_file,
    create_directories,
    save_json,
    load_json,
    save_bin,
    load_bin,
    save_numpy_array_data,
    load_numpy_array_data,
    save_object,
    load_object,
)


def test_write_and_read_yaml(tmp_path):
    yaml_path = tmp_path / "config.yaml"

    data = {
        "model": "RandomForest",
        "n_estimators": 100,
        "learning_rate": 0.1,
    }

    write_yaml_file(yaml_path, data)

    loaded = read_yaml(yaml_path)

    assert loaded.model == "RandomForest"
    assert loaded.n_estimators == 100
    assert loaded.learning_rate == 0.1


def test_create_directories(tmp_path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2" / "nested"

    create_directories([dir1, dir2])

    assert dir1.exists()
    assert dir1.is_dir()

    assert dir2.exists()
    assert dir2.is_dir()


def test_save_and_load_json(tmp_path):
    json_path = tmp_path / "sample.json"

    data = {
        "accuracy": 0.95,
        "f1": 0.88,
    }

    save_json(json_path, data)

    loaded = load_json(json_path)

    assert loaded.accuracy == 0.95
    assert loaded.f1 == 0.88


def test_save_and_load_bin(tmp_path):
    bin_path = tmp_path / "model.joblib"

    obj = {
        "a": 10,
        "b": [1, 2, 3],
    }

    save_bin(obj, bin_path)

    loaded = load_bin(bin_path)

    assert loaded == obj


def test_save_and_load_numpy_array(tmp_path):
    np_path = tmp_path / "array.npy"

    arr = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    save_numpy_array_data(np_path, arr)

    loaded = load_numpy_array_data(np_path)

    assert np.array_equal(arr, loaded)


def test_save_and_load_pickle_object(tmp_path):
    obj_path = tmp_path / "object.pkl"

    obj = {
        "name": "Predictive Maintenance",
        "version": 1,
        "scores": [0.8, 0.9],
    }

    save_object(obj_path, obj)

    loaded = load_object(obj_path)

    assert loaded == obj


def test_write_yaml_replace_existing(tmp_path):
    yaml_path = tmp_path / "replace.yaml"

    write_yaml_file(yaml_path, {"value": 1})

    write_yaml_file(
        yaml_path,
        {"value": 2},
        replace=True,
    )

    loaded = read_yaml(yaml_path)

    assert loaded.value == 2


def test_load_missing_pickle_raises_exception(tmp_path):
    missing = tmp_path / "missing.pkl"

    try:
        load_object(missing)
        assert False, "Expected an exception"
    except Exception:
        assert True