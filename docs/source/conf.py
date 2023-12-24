__doc__ = """Configuration file for the Sphinx documentation builder."""

# -- Project information: -----------------------------------------------------
project = 'scDiffEq'
copyright = '2023, Michael E. Vinyard'
author = 'Michael E. Vinyard'
release = '0.0.53'


# -- General configuration: ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_favicon',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output: ------------------------------------------------
html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']
html_css_files = ['css/custom.css']

html_theme_options = {
    "github_url": "https://github.com/scDiffEq/scDiffEq",
    "twitter_url": "https://twitter.com/vinyard_m",
    "logo": {
      "image_light": "_static/imgs/scdiffeq_logo.png",
      "image_dark": "_static/imgs/scdiffeq_logo.png",
   },
}
autoclass_content = 'init'

favicons = [
    "imgs/scdiffeq.favicon.png",
]