import hashlib
import os
import shutil
import tarfile
from pathlib import Path

import requests
from git import Repo
from tqdm import tqdm


class Downloader:
    """A class to download files from the internet."""

    def download(
        self,
        out_path: str,
        model_name: str,
        model_hash: str,
        url_download: str,
        is_zip: bool = False,
        file_extension: str = None,
    ) -> str:
        """Download a file from the provided URL and store it locally.

        Args:
            model_name (str): Name of the model.
            model_hash (str): Hash of the expected model file.
            url_download (str): URL from where to download the model.

        Returns:
            str: Path to the downloaded model file.
        """
        output_path = Path.home() / ".pixrec" / out_path
        os.makedirs(output_path, exist_ok=True)
        download_file_path = output_path / model_name

        current_model_hash = None

        if download_file_path.exists():
            current_model_hash = self._calculate_hash(download_file_path)

        if model_hash != current_model_hash:
            self._download_file(url_download, download_file_path)

            if is_zip:
                if file_extension == ".tar.gz":
                    with tarfile.open(download_file_path, "r:gz") as tar:
                        tar.extractall(output_path)
                else:
                    shutil.unpack_archive(
                        download_file_path,
                        extract_dir=output_path,
                        format=file_extension,
                    )
                os.remove(download_file_path)

        return download_file_path

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate the MD5 hash of a file.

        Args:
            file_path (Path): Path to the file for which to calculate the hash.

        Returns:
            str: The MD5 hash of the file.
        """
        with file_path.open("rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _download_file(self, url: str, file_path: Path) -> None:
        """Download a file from the provided URL and store it locally.

        Args:
            url (str): URL from where to download the file.
            file_path (Path): Path to save the downloaded file.

        Returns:
            None
        """
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            total_size = int(r.headers.get("Content-Length", 0))

            with file_path.open("wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))


class DownloaderGit:
    """A class to download files from the internet."""

    def download(
        self,
        out_path: str,
        git_repo_url: str,
    ) -> str:
        """Download a file from the provided URL and store it locally.

        Args:
            model_name (str): Name of the model.
            model_hash (str): Hash of the expected model file.
            url_download (str): URL from where to download the model.

        Returns:
            str: Path to the downloaded model file.
        """
        output_path = Path.home() / ".pixrec" / out_path
        if os.path.exists(output_path):
            repo = Repo(output_path)
            origin = repo.remote(name="origin")
            origin.pull()
        else:
            os.makedirs(output_path, exist_ok=True)
            Repo.clone_from(git_repo_url, output_path)

        return output_path
