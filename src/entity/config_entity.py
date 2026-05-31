from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_download_dir: Path
    dataset_name: str