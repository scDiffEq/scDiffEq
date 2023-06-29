# <a href=""><img src=/docs/images/scdiffeq_logo.png alt="scdiffeq_logo" width="320" />

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/scdiffeq.svg)](https://pypi.python.org/pypi/scdiffeq/)
[![PyPI version](https://badge.fury.io/py/scdiffeq.svg)](https://badge.fury.io/py/scdiffeq)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


An analysis framework for modeling dynamical single-cell data with **neural differential equations**, most notably ***stochastic*** differential equations  allow us to build **generative models** of single-cell dynamics.


## Install the development package:

```BASH
git clone https://github.com/mvinyard/sc-neural-diffeqs.git; cd ./sc-neural-diffeqs;

pip install -e .
```

## Main API
  
```python
import scdiffeq as sdq
from neural_diffeqs import NeuralSDE

model = sdq.models.scDiffEq(
    adata, func=NeuralSDE(state_size=50, mu_hidden=[400, 400], sigma_hidden=[400, 400])
)
```
  
```python
model.fit()
```

## Built on:
<img width="50" hspace="20" alt="pytorch_logo" href="https://pytorch.org/" src="https://user-images.githubusercontent.com/47393421/187940001-61655a05-5393-419a-be96-75d11f233d6e.png"><img width="50" href="https://www.pytorchlightning.ai/" hspace="20" alt="pytorch_lightning_logo" src="https://user-images.githubusercontent.com/47393421/187939281-19139d2c-84fe-47b8-a77c-b87e04feca36.png">
<img width="70" href="https://github.com/mvinyard/neural-diffeqs" alt="neural_diffeqs_logo" src="https://user-images.githubusercontent.com/47393421/187945512-c6b9e9e9-92ca-4578-bbbc-f2216727b0e9.png">
