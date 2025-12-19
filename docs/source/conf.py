__doc__ = """Configuration file for the Sphinx documentation builder."""

# -- project info: ------------------------------------------------------------
project = "scdiffeq"
copyright = "2025, Michael E. Vinyard"
author = "Michael E. Vinyard"

import os
import sys
import requests

# Add the docs/source directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config_utils

# Read version from __version__.py
sys.path.insert(0, os.path.abspath("../../"))
from scdiffeq.__version__ import __version__
release = __version__
repository_url = "https://github.com/scDiffEq/scdiffeq-analyses.git"

# -- download reproducibility notebooks ---------------------------------------
class ReproducibilityNotebookPaths(list):
    def __init__(self) -> None:
        self._extensions = [
            "2",
            "3",
            "4",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "s7",
            "s9",
            "s10",
            "s11",
            "s12",
        ]
        for ext in self._extensions:
            self.extend(
                config_utils.fetch_notebook_urls(
                    repository_url=repository_url,
                    path=f"manuscript/figure_{ext}/notebooks/",
                )
            )
notebook_urls = ReproducibilityNotebookPaths()
_ = config_utils.download_notebooks(
    notebook_urls=notebook_urls, destination_dir="./_analyses"
)

# -- download tutorial notebooks ----------------------------------------------
notebook_urls = config_utils.fetch_notebook_urls(repository_url=repository_url, path="tutorials/")
_ = config_utils.download_notebooks(
    notebook_urls=notebook_urls, destination_dir="./_tutorials"
)

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

html_show_sourcelink = False
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
