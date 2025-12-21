__doc__ = """Configuration file for the Sphinx documentation builder."""

# -- import packages: ---------------------------------------------------------
import datetime
import os
import sys

# -- project info: ------------------------------------------------------------
project = "scdiffeq"
current_year = datetime.datetime.now().year
author = "Michael E. Vinyard"
copyright = f"{current_year}, {author}"

# Add the docs/source directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Read version from __version__.py
sys.path.insert(0, os.path.abspath("../../"))
from scdiffeq.__version__ import __version__
release = __version__

# -- download reproducibility notebooks ---------------------------------------
# Notebook downloading strategy:
# - GitHub Actions: Notebooks are copied via shell commands before Sphinx runs
# - ReadTheDocs: Skip downloading (would timeout due to many API calls)
# - Local builds: Download notebooks via GitHub API
on_rtd = os.environ.get("READTHEDOCS") == "True"
on_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"

if on_rtd:
    print("ReadTheDocs build detected. Skipping notebook downloads.")
elif on_github_actions:
    print("GitHub Actions build detected. Notebooks already copied via workflow.")
else:
    # Only download notebooks for local builds
    try:
        import config_utils
        repository_url = "https://github.com/scDiffEq/scdiffeq-analyses.git"
        
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
                    try:
                        urls = config_utils.fetch_notebook_urls(
                            repository_url=repository_url,
                            path=f"manuscript/figure_{ext}/notebooks/",
                        )
                        self.extend(urls)
                    except Exception as e:
                        print(f"Warning: Failed to fetch notebook URLs for figure_{ext}: {e}")
                        continue
        
        notebook_urls = ReproducibilityNotebookPaths()
        if notebook_urls:
            try:
                _ = config_utils.download_notebooks(
                    notebook_urls=notebook_urls, destination_dir="./_analyses"
                )
            except Exception as e:
                print(f"Warning: Failed to download reproducibility notebooks: {e}")
        else:
            print("Warning: No reproducibility notebook URLs found. Skipping download.")
        
        # -- download tutorial notebooks ----------------------------------------------
        try:
            tutorial_urls = config_utils.fetch_notebook_urls(
                repository_url=repository_url, path="tutorials/"
            )
            if tutorial_urls:
                try:
                    _ = config_utils.download_notebooks(
                        notebook_urls=tutorial_urls, destination_dir="./_tutorials"
                    )
                except Exception as e:
                    print(f"Warning: Failed to download tutorial notebooks: {e}")
            else:
                print("Warning: No tutorial notebook URLs found. Skipping download.")
        except Exception as e:
            print(f"Warning: Failed to fetch tutorial notebook URLs: {e}")

    except Exception as e:
        print(f"Warning: Notebook downloading disabled due to error: {e}")

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

# Include .nojekyll for GitHub Pages (prevents ignoring _static, _analyses folders)
html_extra_path = [".nojekyll"]

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
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'anndata': ('https://anndata.readthedocs.io/en/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
}

# Add settings for better type hints display
python_use_unqualified_type_names = True

# Keep your autodoc class content setting
autoclass_content = "both"  # Changed from "init" to "both" to show both class and __init__ docstrings

# Suppress warnings for missing references to external types
# These are common type hints that don't resolve well with intersphinx
nitpick_ignore = [
    ('py:class', 'optional'),
    ('py:class', 'None'),
    ('py:class', '='),
    ('py:class', 'obj'),
    ('py:class', 'diffeq'),
    ('py:class', 'Second-most central function'),
    ('py:class', 'this autoparsing base'),
    ('py:class', 'GroupedMetrics'),
    ('py:class', 'LightningCheckpoint'),
]

nitpick_ignore_regex = [
    # Ignore private/internal class references
    (r'py:class', r'.*\._.*'),
    # Ignore pandas internal types
    (r'py:class', r'pandas\.core\..*'),
    # Ignore anndata internal types  
    (r'py:class', r'anndata\._core\..*'),
    # Ignore matplotlib internal types
    (r'py:class', r'matplotlib\.axes\._axes\..*'),
    # Ignore lightning internal types
    (r'py:class', r'lightning\.pytorch\.core\..*'),
    # Ignore voyager types
    (r'py:class', r'voyager\..*'),
    # Ignore ABCParse internal types
    (r'py:class', r'ABCParse\..*'),
    # Common shorthand type hints
    (r'py:class', r'pd\..*'),
    (r'py:class', r'np\..*'),
    (r'py:class', r'plt\..*'),
    (r'py:class', r'anndata\.AnnData'),
    (r'py:class', r'lightning\.LightningModule'),
]

# Suppress common warnings that don't affect documentation quality
suppress_warnings = [
    'autosectionlabel.*',
    'ref.python',  # Suppress "more than one target found" warnings
    'ref.class',   # Suppress class reference warnings
]

favicons = [{"rel": "icon", "href": "scdiffeq.favicon.png"}]
