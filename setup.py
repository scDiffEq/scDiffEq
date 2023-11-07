
# -- import packages: ---------------------------------------------------------
import setuptools
import re
import os
import sys


# -- fetch requirements packages: ---------------------------------------------
with open('requirements.txt') as f:
    requirements = f.read().splitlines()


# -- run setup: ---------------------------------------------------------------
setuptools.setup(
    name="scdiffeq",
    version="0.0.48rc2",
    python_requires=">3.9.0",
    author="Michael E. Vinyard",
    author_email="mvinyard.ai@gmail.com",
    url="https://github.com/scDiffEq/scDiffEq",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description="scDiffEq: modeling single-cell dynamics using neural differential equations.",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="MIT",
)
