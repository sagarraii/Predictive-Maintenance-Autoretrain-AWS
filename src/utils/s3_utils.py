from pathlib import Path
import subprocess


class S3Sync:
    @staticmethod
    def upload_folder(local_folder: str, s3_uri: str):
        subprocess.run(
            [
                "aws",
                "s3",
                "sync",
                local_folder,
                s3_uri,
                "--delete",
            ],
            check=True,
        )

    @staticmethod
    def download_folder(s3_uri: str, local_folder: str):
        Path(local_folder).mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "aws",
                "s3",
                "sync",
                s3_uri,
                local_folder,
            ],
            check=True,
        )