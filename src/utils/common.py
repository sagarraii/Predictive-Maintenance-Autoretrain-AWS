import sys
import json
import yaml
import joblib
import pickle

import numpy as np

from pathlib import Path
from typing import Any

from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from src.logger.logging_config import logger
from src.exception.custom_exception import CustomException

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns a ConfigBox."""

    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("YAML file is empty")
    
    except Exception as e:
        raise CustomException(e, sys)

# @ensure_annotations
def write_yaml_file(file_path: Path, content: Any, replace: bool = False) -> None:
    """Writes content to a YAML file."""

    try:
        if replace and file_path.exists():
            file_path.unlink()  # Modern pathlib deletion
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)

        logger.info(f"YAML file saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

# @ensure_annotations
def create_directories(path_to_directories: list[Path], verbose: bool = True) -> None:
    """Creates a list of directories."""

    try:
        for path in path_to_directories:
            path.mkdir(parents=True, exist_ok=True)

            if verbose:
                logger.info(f"Created directory at: {path}")

    except Exception as e:
        raise CustomException(e, sys)

@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """Saves dictionary data to a JSON file."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        logger.info(f"JSON file saved at: {path}")

    except Exception as e:
        raise CustomException(e, sys)

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads JSON file data as class attributes."""

    try:
        with open(path, "r") as f:
            content = json.load(f)

        logger.info(f"JSON file loaded successfully from: {path}")
        return ConfigBox(content)
    
    except Exception as e:
        raise CustomException(e, sys)

@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    """Saves binary file using joblib."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(value=data, filename=path)

        logger.info(f"Binary file saved at: {path}")

    except Exception as e:
        raise CustomException(e, sys)

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads binary data using joblib."""

    try:
        data = joblib.load(path)

        logger.info(f"Binary file loaded from: {path}")
        return data
    
    except Exception as e:
        raise CustomException(e, sys)

# @ensure_annotations
def save_numpy_array_data(file_path: Path, array: np.ndarray) -> None:
    """Saves numpy array data to a file."""

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

        logger.info(f"Numpy array saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

@ensure_annotations
def load_numpy_array_data(file_path: Path) -> np.ndarray:
    """Loads numpy array data from a file."""

    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)

#@ensure_annotations obj: Any(accepting generic object)
def save_object(file_path: Path, obj: Any) -> None:
    """Pickles a python object using standard serialization."""

    try:
        logger.info(f"Saving pickle object to: {file_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

#@ensure_annotations
def load_object(file_path: Path) -> Any:
    """Loads a pickled object from a file path."""

    try:
        if not file_path.exists():
            raise FileNotFoundError(f"The file: {file_path} does not exist.")
        
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
