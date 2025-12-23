# <a href=""><img src=/docs/images/scdiffeq_logo.png alt="scdiffeq_logo" width="320" />

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/scdiffeq.svg)](https://pypi.python.org/pypi/scdiffeq/)
[![PyPI version](https://badge.fury.io/py/scdiffeq.svg)](https://badge.fury.io/py/scdiffeq)
<a href="https://doi.org/10.5281/zenodo.17238593"><img src="https://zenodo.org/badge/616276336.svg" alt="DOI"></a>

An analysis framework for modeling dynamical single-cell data with **neural differential equations**, most notably ***stochastic*** differential equations  allow us to build **generative models** of single-cell dynamics.

## Quickstart

Please see the [**scDiffEq website**](https://scdiffeq.com) for a quickstart notebook: [link](https://www.scdiffeq.com/_tutorials/quickstart.html)

## Install the development package

Install generally only takes a few seconds.

### Using uv (recommended)

```BASH
git clone https://github.com/scDiffEq/scDiffEq.git; cd ./scDiffEq;

# Install uv if you haven't already: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Using pip

```BASH
git clone https://github.com/scDiffEq/scDiffEq.git; cd ./scDiffEq;

pip install -e .
```

### With documentation dependencies

```BASH
# Using uv
uv sync --extra docs

# Using pip
pip install -e ".[docs]"
```

## Main API
  
```python
import scdiffeq as sdq

model = sdq.scDiffEq(adata=adata)

model.fit(train_epochs = 1500)
```

## Built on

<img width="50" hspace="20" alt="pytorch_logo" href="https://pytorch.org/" src="https://user-images.githubusercontent.com/47393421/187940001-61655a05-5393-419a-be96-75d11f233d6e.png"><img width="50" href="https://www.pytorchlightning.ai/" hspace="20" alt="pytorch_lightning_logo" src="https://user-images.githubusercontent.com/47393421/187939281-19139d2c-84fe-47b8-a77c-b87e04feca36.png">
<img width="110" href="https://github.com/mvinyard/neural-diffeqs" alt="neural_diffeqs_logo" src="https://github.com/user-attachments/assets/e9cc2860-bfa4-43a1-bc15-85d074b44fc7" />

## System requirements

- Developed on linux20.04 and MacOS (with Apple Silicon), using Python3.11.
- Software dependencies are listed in [pyproject.toml](./pyproject.toml).
- Tested with NVIDIA GPUs (A100, T4) and Apple Silicon. Most datasets likely only require an NVIDIA Tesla T4 (free in Google Colab).

## Reproducibility

- All results described in the [manuscript](https://rdcu.be/eVhnL) detailing scDiffEq can be reproduced using notebooks in the companion repository: [scdiffeq-analyses](https://github.com/scDiffEq/scdiffeq-analyses)
