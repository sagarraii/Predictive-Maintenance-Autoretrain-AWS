from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_download_dir: Path
    dataset_name: str

@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    data_file_path: Path
    all_schema: dict