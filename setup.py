from setuptools import setup
import re
import os
import sys

setup(
    name="scdiffeq",
    version="0.0.45",
    python_requires=">3.7.0",
    author="Michael E. Vinyard - Harvard University - Massachussetts General Hospital - Broad Institute of MIT and Harvard",
    author_email="mvinyard@broadinstitute.org",
    url="https://github.com/mvinyard/sc-neural-diffeqs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description="scDiffEq: modelling single-cell dynamics using neural differential equations.",
    packages=setuptools.find_packages(),
    install_requires=[
        "anndata>=0.8",
        "torch>=1.12.0",
        "pykeops>=2.1",
        "geomloss>=0.2.5",
        "licorice_font>=0.0.3",
        "torchdiffeq>=0.2.3",
        "torchsde>=0.2.5",
        "scikit-learn>=1.0.2",
        "umap-learn>=0.5.2",
        "pytorch-lightning>=1.7.7",
        "neural-diffeqs==0.2.0rc0",
        "torch-nets>=0.0.2",
        "torch-adata",
        "autodevice>=0.0.2",
        "brownian-diffuser>=0.0.1",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="MIT",
)
