# <a href=""><img src=/docs/images/scdiffeq_logo.png alt="scdiffeq_logo" width="320" />

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/scdiffeq.svg)](https://pypi.python.org/pypi/scdiffeq/)
[![PyPI version](https://badge.fury.io/py/scdiffeq.svg)](https://badge.fury.io/py/scdiffeq)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/scdiffeq/badge/?version=latest)](https://docs.scdiffeq.com/en/latest/?badge=latest)


An analysis framework for modeling dynamical single-cell data with **neural differential equations**, most notably ***stochastic*** differential equations  allow us to build **generative models** of single-cell dynamics.

## Quickstart

Please see the [**scDiffEq website**](https://docs.scdiffeq.com/en/latest/index.html) for a quickstart notebook: [link](https://docs.scdiffeq.com/en/latest/_notebooks/quickstart.html)

## Install the development package:

Install generally only takes a few seconds.

```BASH
git clone https://github.com/mvinyard/sc-neural-diffeqs.git; cd ./sc-neural-diffeqs;

pip install -e .
```

## Main API
  
```python
import scdiffeq as sdq

model = sdq.scDiffEq(
    adata=adata, potential_type="fixed", train_lr=1e-4, train_step_size=1200
)
model.fit(train_epochs = 1500)
```
  

## Built on:
<img width="50" hspace="20" alt="pytorch_logo" href="https://pytorch.org/" src="https://user-images.githubusercontent.com/47393421/187940001-61655a05-5393-419a-be96-75d11f233d6e.png"><img width="50" href="https://www.pytorchlightning.ai/" hspace="20" alt="pytorch_lightning_logo" src="https://user-images.githubusercontent.com/47393421/187939281-19139d2c-84fe-47b8-a77c-b87e04feca36.png">
<img width="70" href="https://github.com/mvinyard/neural-diffeqs" alt="neural_diffeqs_logo" src="https://user-images.githubusercontent.com/47393421/187945512-c6b9e9e9-92ca-4578-bbbc-f2216727b0e9.png">


## System requirements
- Developed on linux20.04 and MacOS (with Apple Silicon), using Python3.11.
- Software dependencies are listed in [requirements.txt](./requirements.txt)
- Tested with NVIDIA GPUs (A100, T4) and Apple Silicon. Most datasets likely only require an NVIDIA Tesla T4 (free in Google Colab).

## Reproducibiliuty
- All results described in the [manuscript](https://www.biorxiv.org/content/10.1101/2023.12.06.570508v2) detailing scDiffEq can be reproduced using notebooks in the companion repository: [scdiffeq-analyses](https://github.com/scDiffEq/scdiffeq-analyses)
