__doc__ = """Configuration file for the Sphinx documentation builder."""

# -- project info: ------------------------------------------------------------
project = "scdiffeq"
copyright = "2024, Michael E. Vinyard"
author = "Michael E. Vinyard"

# Read version from __version__.py
import os
import sys
sys.path.insert(0, os.path.abspath("../../"))
from scdiffeq.__version__ import __version__
release = __version__

# -- config: ------------------------------------------------------------------
import requests
import time
from typing import List, Optional, get_type_hints

# -- new nb fetch: ------------------------------------------------------------
class NotebookURLs:
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize with optional GitHub token for authenticated requests.
        
        Args:
            github_token: Personal access token from GitHub
        """
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        if self.github_token:
            self.headers['Authorization'] = f'token {self.github_token}'
        
        # Add rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests

    def _URL_factory(self, path: str) -> str:
        """
        Generate GitHub API URL for content listing.
        
        Args:
            path: Repository path (example: manuscript/Figure2)
        """
        return f"https://api.github.com/repos/scDiffEq/scdiffeq-analyses/contents/{path}/notebooks?ref=main"

    def _handle_rate_limit(self, response: requests.Response) -> None:
        """Handle rate limit information from GitHub response."""
        remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
        
        if remaining == 0:
            wait_time = reset_time - time.time()
            if wait_time > 0:
                print(f"Rate limit exceeded. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

    def _fetch(self, url: str) -> List[str]:
        """
        Fetch notebook URLs with retry logic and rate limit handling.
        
        Args:
            url: GitHub API URL to fetch from
        """
        # Implement request rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        max_retries = 3
        for attempt in range(max_retries):
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
                print(f"Failed to fetch from GitHub after {max_retries} attempts: {str(e)}")
                return []
        
        return []

    def __call__(self) -> List[str]:
        """Fetch all notebook URLs with progress tracking."""
        paths = []
        fig_nums = ["2", "3", "4", "s1", "s2", "s3", "s4", "s5", "s7", "s9", "s10", "s11", "s12"]
        
        for i, fn in enumerate(fig_nums, 1):
            print(f"Fetching figure {fn} ({i}/{len(fig_nums)})...")
            fig_paths = self._fetch(self._URL_factory(f"manuscript/figure_{fn}"))
            paths.extend(fig_paths)
            
        return paths

def download_notebooks():
    """Download notebooks with authentication and error handling."""
    try:
        github_token = os.environ.get('GITHUB_TOKEN')
        url_fetcher = NotebookURLs(github_token)
        notebook_urls = url_fetcher()
        
        if not notebook_urls:
            print("Warning: No notebook URLs were found")
            return
            
        os.makedirs("./_notebooks", exist_ok=True)
        
        # Create headers specifically for raw.githubusercontent.com
        headers = {
            'Accept': 'application/vnd.github.v3.raw',
            'Authorization': f'Bearer {github_token}' if github_token else None,
            'User-Agent': 'scdiffeq-docs'
        }
        headers = {k: v for k, v in headers.items() if v is not None}
        
        print(f"\nDownloading {len(notebook_urls)} notebooks...")
        for i, url in enumerate(notebook_urls, 1):
            try:
                print(f"Downloading {os.path.basename(url)} ({i}/{len(notebook_urls)})...")
                # Convert raw URL to API URL format
                api_url = url.replace('raw.githubusercontent.com', 'api.github.com/repos').replace('/main/', '/contents/')
                r = requests.get(api_url, headers=headers, timeout=30)
                
                if r.status_code == 200:
                    content = r.content
                    if r.headers.get('content-type') == 'application/json':
                        # If we got JSON instead of raw content, extract the content
                        content = requests.get(r.json()['download_url'], headers=headers, timeout=30).content
                    
                    with open(os.path.join("./_notebooks", os.path.basename(url)), "wb") as f:
                        f.write(content)
                else:
                    print(f"Error status {r.status_code} for {url}: {r.text}")
                    
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
                continue
                
        print("Download complete!")
        
    except Exception as e:
        print(f"Error in download_notebooks: {str(e)}")

# Run the notebook download
download_notebooks()

# -- Your existing path setup -------------------------------------------------
sys.path.insert(0, os.path.abspath("../../"))

# -- Add autodoc settings -----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# This helps with showing the correct parameter types
autodoc_type_aliases = {
    'ArrayLike': 'numpy.typing.ArrayLike',
    # Add any other type aliases your project uses
}

# Ensures that your docstrings are properly parsed
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Your existing extensions list --
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",  # Add this for external references
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_favicon",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- html output options: -----------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_theme_options = {
    "github_url": "https://github.com/scDiffEq/scDiffEq",
    "twitter_url": "https://twitter.com/vinyard_m",
    "logo": {
        "image_light": "scdiffeq_logo.png",
        "image_dark": "scdiffeq_logo.dark_mode.png",
    },
}

# Add intersphinx mapping for external package references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    # Add other relevant packages your project depends on
}

# Add settings for better type hints display
python_use_unqualified_type_names = True

# Keep your autodoc class content setting
autoclass_content = "both"  # Changed from "init" to "both" to show both class and __init__ docstrings

favicons = [{"rel": "icon", "href": "scdiffeq.favicon.png"}]
