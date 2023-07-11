import setuptools
import re
import os
import sys

setuptools.setup(
    name="scdiffeq",
    version="0.0.48rc0",
    python_requires=">3.9.0",
    author="Michael E. Vinyard - Harvard University - Massachussetts General Hospital - Broad Institute of MIT and Harvard",
    author_email="mvinyard@broadinstitute.org",
    url="https://github.com/mvinyard/sc-neural-diffeqs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description="scDiffEq: modelling single-cell dynamics using neural differential equations.",
    packages=setuptools.find_packages(),
    install_requires=[
        "anndata>=0.9.1",
        "torch>=2.0.0",
        "pykeops>=2.1",
        "geomloss>=0.2.5",
        "licorice_font>=0.0.3",
        "torchsde>=0.2.5",
        "scikit-learn>=1.0.2",
        "umap-learn>=0.5.3",
        "lightning>=2.0.1",
        "neural-diffeqs==0.3.1rc0",
        "torch-nets>=0.0.4",
        "torch-adata>=0.0.23",
        "autodevice>=0.0.2",
        "brownian-diffuser>=0.0.2",
        "vinplots>=0.0.75",
        "annoyance==0.0.18",
        "ABCParse==0.0.3",
        "scdiffeq-plots==0.0.1rc2",
        "plotly==5.15.0",
        "scvelo==0.2.5",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="MIT",
)
