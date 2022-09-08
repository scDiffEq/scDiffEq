from setuptools import setup
import re
import os
import sys

setup(
    name="scdiffeq",
    version="0.0.43",
    python_requires=">3.7.0",
    author="Michael E. Vinyard - Harvard University - Massachussetts General Hospital - Broad Institute of MIT and Harvard",
    author_email="mvinyard@broadinstitute.org",
    url="https://github.com/mvinyard/sc-neural-diffeqs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description="scDiffEq: modelling single-cell dynamics using neural differential equations.",
    packages=[
        "scdiffeq",
        "scdiffeq._dev_utils",
        "scdiffeq._io",
        "scdiffeq._models",
        "scdiffeq._models._core",
        "scdiffeq._models._core._base_ancilliary",
        "scdiffeq._models._core._lightning_callbacks",
        "scdiffeq._plotting",
        "scdiffeq._preprocessing",
        "scdiffeq._tools",
    ],
    install_requires=[
        "anndata>=0.8",
	"torch>=1.12.0",
        "geomloss>=0.2.5",
	"pydk>=0.0.51",
        "licorice_font>=0.0.3",
	"neural-diffeqs>=0.0.31",
        "torchdiffeq>=0.2.1",
        "torchsde>=0.2.5",
	"torch-adata>=0.0.2",
        "scikit-learn>=1.0.2",
        "umap-learn>=0.5.2",
	"pytorch_lightning>=1.7.5",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="MIT",
)
