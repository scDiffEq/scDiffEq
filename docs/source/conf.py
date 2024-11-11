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

            # Filter for .ipynb files and print their download URLs
            ipynb_files = [
                file["download_url"]
                for file in files
                if file["name"].endswith(".ipynb")
            ]
            return ipynb_files
            # for file_url in ipynb_files:
            #     print(file_url)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    @property
    def Figure2(self):
        return self._fetch(self._URL_factory("manuscript/figure_2"))

    @property
    def Figure3(self):
        return self._fetch(self._URL_factory("manuscript/figure_3"))

    @property
    def Figure4(self):
        return self._fetch(self._URL_factory("manuscript/figure_4"))
    
    @property
    def FigureS1(self):
        return self._fetch(self._URL_factory("manuscript/figure_s1"))
    
    @property
    def FigureS2(self):
        return self._fetch(self._URL_factory("manuscript/figure_s2"))
    
    @property
    def FigureS3(self):
        return self._fetch(self._URL_factory("manuscript/figure_s1"))
    
    @property
    def FigureS4(self):
        return self._fetch(self._URL_factory("manuscript/figure_s2"))

    def __call__(self):

        paths = self.Figure2 + self.Figure3 + self.Figure4 + self.FigureS1 + self.FigureS2 + self.FigureS3 + self.FigureS4
        return paths


# ------------------------------------------------


def download_notebooks():
    url_fetcher = NotebookURLs()
    notebook_urls = url_fetcher()
    os.makedirs("./_notebooks", exist_ok=True)  # Ensure the target directory exists
    for url in notebook_urls:
        r = requests.get(url)
        with open(os.path.join("./_notebooks", os.path.basename(url)), "wb") as f:
            f.write(r.content)


download_notebooks()

sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
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
autoclass_content = "init"

favicons = [{"rel": "icon", "href": "scdiffeq.favicon.png"}]

# -- notes: -------------------------------------------------------------------
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
