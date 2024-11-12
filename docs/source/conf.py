__doc__ = """Configuration file for the Sphinx documentation builder."""

# -- project info: ------------------------------------------------------------
project = "scdiffeq"
copyright = "2023, Michael E. Vinyard"
author = "Michael E. Vinyard"
release = "0.1.0"


# -- config: ------------------------------------------------------------------
import os
import sys
import requests

# -- set typing: --------------------------------------------------------------
import typing
from typing import get_type_hints

# -- new nb fetch: ------------------------------------------------------------
class NotebookURLs:
    def __init__(self): ...

    def _URL_factory(self, path: str):
        """example: manuscript/Figure2"""
        return f"https://api.github.com/repos/scDiffEq/scdiffeq-analyses/contents/{path}/notebooks?ref=main"

    def _fetch(self, url: str):
        response = requests.get(url)

        if response.status_code == 200:
            files = response.json()

            ipynb_files = [
                file["download_url"]
                for file in files
                if file["name"].endswith(".ipynb")
            ]
            return ipynb_files
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    def __call__(self):
        
        paths = []
        
        fig_nums = ["2", "3", "4", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
        for fn in fig_nums:
            paths += self._fetch(self._URL_factory(f"manuscript/figure_{fn}"))
        return paths

# -----------------------------------------------------------------------------

def download_notebooks():
    url_fetcher = NotebookURLs()
    notebook_urls = url_fetcher()
    os.makedirs("./_notebooks", exist_ok=True)  # Ensure the target directory exists
    for url in notebook_urls:
        r = requests.get(url)
        with open(os.path.join("./_notebooks", os.path.basename(url)), "wb") as f:
            f.write(r.content)


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
