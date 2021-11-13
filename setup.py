from setuptools import setup
import re
import os
import sys

setup(
    name="scdiffeq",
    python_requires=">3.8.0",
    author="Michael E. Vinyard - Harvard University - Massachussetts General Hospital - Broad Institute of MIT and Harvard",
    author_email="mvinyard@broadinstitute.org",
    url="https://github.com/pinellolab/sc-neural-ODEs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description="scdiffeq: single cell dynamics using neural ODEs and variations thereof.",
    packages=[
        "scdiffeq",
        "scdiffeq._tools",
        "scdiffeq._plotting",
        "scdiffeq._utilities",
        "scdiffeq._data",
    ],
    install_requires=[
        anndata,
        sklearn,
        matplotlib,
        anndata,
        "torch>=1.1.0",
        "numpy>=1.19.2",
        "pandas>=1.1.2",
        "torchdiffeq>=0.2.1",
        "nb-black>=1.0.7",
        "geomloss>=0.2.4",
        "torchsde",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="MIT",
)
