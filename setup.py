from setuptools import setup
import re
import os
import sys

setup(
    name="scdiffeq",
    python_requires=">3.7.0",
    author="Michael E. Vinyard - Harvard University - Massachussetts General Hospital - Broad Institute of MIT and Harvard",
    author_email="mvinyard@broadinstitute.org",
    url="https://github.com/pinellolab/sc-neural-ODEs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description="scDiffEq: single cell dynamics using neural ODEs.",
    packages=[
        "scdiffeq",
        "scdiffeq._io",
        "scdiffeq._models",
	"scdiffeq._plotting",
	"scdiffeq._tools",
    ],
    install_requires=[
        "anndata>=0.7.8",
        "geomloss>=0.2.4",
        "matplotlib>=3.5.0",
        "pandas>=1.3.5",
	"pydk>=0.0.51",
        "licorice_font>=0.0.3",
        "torch>=1.1.0",
        "torchdiffeq>=0.2.1",
        "torchsde>=0.2.5",
        "scikit-learn>=1.0.2",
        "umap-learn>=0.5.2",
	"pytorch_lightning>=1.6.5",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="MIT",
)
