import logging
import os
import requests
from typing import List, Union


logger = logging.getLogger(__name__)


class NotebookDownloader:
    """Download notebooks with authentication and error handling."""

    def __init__(
        self,
        destination_dir: str,
        user_agent: str = "scdiffeq-docs",
    ) -> None:

        self._user_agent = user_agent
        self.destination_dir = destination_dir
        os.makedirs(self.destination_dir, exist_ok=True)

    @property
    def GitHub_token(self) -> str:
        return os.environ.get("GITHUB_TOKEN")

    @property
    def headers(self):
        if not hasattr(self, "_headers"):
            _headers = {
                "Accept": "application/vnd.github.v3.raw",
                "Authorization": (
                    f"Bearer {self.GitHub_token}" if self.GitHub_token else None
                ),
                "User-Agent": self._user_agent,
            }
            self._headers = {k: v for k, v in _headers.items() if v is not None}
        return self._headers

    def _forward(self, url):
        logger.info(f"Downloading {os.path.basename(url)} ({self._i}/{self._n_nbs})...")
        # Convert raw URL to API URL format
        api_url = url.replace(
            "raw.githubusercontent.com", "api.github.com/repos"
        ).replace("/main/", "/contents/")
        r = requests.get(api_url, headers=self.headers, timeout=30)
        if r.status_code == 200:
            content = r.content
            if r.headers.get("content-type") == "application/json":
                # If we got JSON instead of raw content, extract the content
                content = requests.get(
                    r.json()["download_url"],
                    headers=self.headers,
                    timeout=30,
                ).content

            fpath = os.path.join(self.destination_dir, os.path.basename(url))
            with open(fpath, "wb") as f:
                f.write(content)
            return fpath
        else:
            print(f"Error status {r.status_code} for {url}: {r.text}")

    def forward(self, notebook_urls: List[str]):
        self._n_nbs = len(notebook_urls)
        logger.info(f"\nDownloading {self._n_nbs} notebooks...")
        fpaths = []
        for i, url in enumerate(notebook_urls, 1):
            self._i = i
            try:
                fpath = self._forward(url=url)
                fpaths.append(fpath)
            except Exception as e:
                logger.error(f"Error downloading {url}: {str(e)}")
                continue
        logger.info("Download complete.")
        return fpaths

    def __call__(self, notebook_urls, *args, **kwargs) -> Union[List[str], None]:
        try:
            return self.forward(notebook_urls)
        except Exception as e:
            message = f"Error in download_reproducibility_notebooks: {str(e)}"
            logger.error(message)


def download_notebooks(notebook_urls: List[str], destination_dir: str) -> None:
    """Download notebooks with authentication and error handling."""
    try:
        downloader = NotebookDownloader(destination_dir=destination_dir)
        return downloader(notebook_urls=notebook_urls)
    except Exception as e:
        logger.error(f"Error in download_notebooks: {str(e)}")
