# -- import packages: ---------------------------------------------------------
import setuptools
import re
import os
import sys

# -- constants: --------------------------------------------------------------
name = "scdiffeq"

# -- fetch requirements packages: ---------------------------------------------
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open(f"{name}/__version__.py") as v:
    exec(v.read())


# -- run setup: ---------------------------------------------------------------
setuptools.setup(
    name=name,
    version=__version__,
    python_requires=">3.10",
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
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="MIT",
)
