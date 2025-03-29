# -- import packages: ---------------------------------------------------------
import os
import time
import requests

# -- set type hints: ----------------------------------------------------------
from typing import Dict, List, Optional

# -- operational cls: ---------------------------------------------------------
class GitHubNotebookURLFetcher:
    def __init__(
        self,
        repository_url: str,
        github_token: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize with optional GitHub token for authenticated requests.

        Args:
            github_token: Personal access token from GitHub
        """

        self.repository_url = repository_url
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"

        # -- rate limiting: ---------
        self._max_retries = max_retries
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests

    @property
    def repository(self) -> Dict[str, str]:
        if not hasattr(self, "_repository"):
            _split_repo_url = self.repository_url.split("/")
            self._repository = {
                "name": _split_repo_url[-1].split(".git")[0],
                "org": _split_repo_url[-2],
            }
        return self._repository

    def _URL_factory(self, path: str) -> str:
        """
        Generate GitHub API URL for content listing.

        Args:
            path: Repository path (example: manuscript/Figure2)
        """
        return f"https://api.github.com/repos/{self.repository['org']}/{self.repository['name']}/contents/{path}?ref=main"

    def _handle_rate_limit(self, response: requests.Response) -> None:
        """Handle rate limit information from GitHub response."""
        remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))

        if remaining == 0:
            wait_time = reset_time - time.time()
            if wait_time > 0:
                print(f"Rate limit exceeded. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

    def _fetch(self, path: str) -> List[str]:
        """
        Fetch notebook URLs with retry logic and rate limit handling.

        Args:
            path (str):
        """

        # url: GitHub API URL to fetch from
        url = self._URL_factory(path=path)

        # Implement request rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        for attempt in range(self._max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                self.last_request_time = time.time()

                if response.status_code == 200:
                    files = response.json()
                    return [
                        file["download_url"]
                        for file in files
                        if file["name"].endswith(".ipynb")
                    ]
                elif response.status_code == 403:
                    self._handle_rate_limit(response)
                    if attempt < max_retries - 1:
                        continue
                    print(f"GitHub API access forbidden. Response: {response.text}")
                    return []
                elif response.status_code == 404:
                    print(f"Warning: Path not found: {url}")
                    return []
                else:
                    print(f"GitHub API error {response.status_code}: {response.text}")
                    return []

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Request failed, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                print(
                    f"Failed to fetch from GitHub after {self._max_retries} attempts: {str(e)}"
                )
                return []

        return []

    def __call__(self, path: str) -> List[str]:
        """Fetch all notebook URLs with progress tracking."""
        return self._fetch(path=path)


def fetch_notebook_urls(repository_url: str, path: str) -> List[str]:
    """Fetch notebook URLs from a GitHub repository path.

    Args:
        repository_url: str
            The URL of the GitHub repository to fetch notebooks from.
        path: str 
            The path within the repository to search for notebooks.

    Returns:
        List[str]: A list of URLs for the notebook files found at the specified path.
    """
    fetcher = GitHubNotebookURLFetcher(repository_url=repository_url)
    return fetcher(path=path)
