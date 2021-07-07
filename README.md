# Single-Cell Neural Differential Equations

![scdiffeq_logo](https://i.imgur.com/EOJ6W9R.png)

* <a href="https://docs.google.com/document/d/1gCAEqp0lZsxt3LqhKeaeSIIqthrAg3IKkxWXP1sOvgs/edit#">**Manuscript in progress**</a>
* <a href="https://github.com/pinellolab/sc-neural-ODEs/notebooks">**Example notebooks**</a>
* <a href="https://github.com/pinellolab/sc-neural-ODEs/scdiffeq">**scdiffeq**</a> (development package)

## Install the development package:
```
git clone git@github.com:pinellolab/sc-neural-ODEs.git

pip install -e .
```

## Tutorials
* <a href="">**10x Genomics PBMC Tutorial**</a> (in progress)
* <a href="">**Time-resolved, lineage-tracing data tutorial**</a> (in progress, LARRY downsample)

## Method inputs and outputs
* **Minimum input**: cell x gene expression table
* **Output**:


## Usage principles

 #### 1. Data import 


```python
# load AnnData GEX object from standard preprocessing such as Scanpy
adata = sdq.data.load_LARRY()
```

> AnnData object with n_obs × n_vars = 130887 × 25289
> 
>$\:\:\:\:\:\:\:\:\:$obs:$\:$ 'n_counts', 'Time_Point', 'Source', 'Well', 'mito_frac', 'time_info', 'state_info'
$\:\:\:\:\:\:\:\:\:$var:$\:$ 'highly_variable'
$\:\:\:\:\:\:\:\:\:$uns:$\:$ 'clonal_time_points', 'data_des', 'max_mito', 'min_tot', 'neighbors', 
$\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$'state_info_colors', 'time_info_colors', 'time_ordering', 'umap'
$\:\:\:\:\:\:\:\:\:$obsm:$\:$ 'X_clone', 'X_emb', 'X_pca', 'X_umap'

#### 2. Standard preprocessing 

```python
sdq.ut.preprocess(adata)
```

#### 3. Define a torch neural network as an ODE

In this case, a stochastic differential equation (SDE), an evolution of the original neural differential equation is the neural framework we will train. As we demonstrate in our manuscript (Figure N), SDEs are superior to NDEs in describing GEX data. The SDE has two main components:

(**1**) drift *describes *

(**2**) diffusion

> $$ \frac{\partial (x,t)}{\partial t} = \nabla[ p(x,t) \nabla F(x)] + D\nabla^2 p(x,t) + R(x) p(x,t), $$

This framework for mathematical modelling of gene expression data is actually somewhat analagous to Fokker-Planck equation (i.e., Population Balance equation) that employed in PBA from *Weinreb and Klein* (2018).


```python
# example SDE taken from https://github.com/google-research/torchsde

class SDE(torch.nn.Module):
    noise_type, sde_type = 'general', 'ito'
    
    def __init__(self):
        super().__init__()
        self.mu = torch.nn.Linear(state_size, 
                                  state_size)
        self.sigma = torch.nn.Linear(state_size, 
                                     state_size * brownian_size)

    # Drift
    def f(self, t, y):
        return self.mu(y)  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        return self.sigma(y).view(batch_size, 
                                  state_size, 
                                  brownian_size)

```

```python
adata.uns['diffeq']
```
>sde_func(
$\:\:\:\:\:\:\:\:\:\:\:\:$ (position_net_drift): Sequential(
    $\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$(0): Linear(in_features=2, out_features=10, bias=True)
    $\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$(1): ReLU()
    $\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$(2): Linear(in_features=10, out_features=2, bias=True)
    $\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$(3): ReLU()
    $\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$(4): Linear(in_features=2, out_features=10, bias=True)
    $\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$(5): ReLU()
    $\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$(6): Linear(in_features=10, out_features=2, bias=True)
    $\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$(7): ReLU()
    $\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$(8): Linear(in_features=2, out_features=10, bias=True)
    $\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$(9): ReLU()
    $\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$(10): Linear(in_features=10, out_features=2, bias=True)
$\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$  )
$\:\:\:\:\:\:\:\:\:\:\:\:$  (position_net_diff): Sequential(
$\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$ (0): Linear(in_features=2, 
$\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$ out_features=2, bias=True)
$\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$  )
$\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:\:$)

#### 4. Learn the dynamical gene expression on a training set of data

This is the key algorithmic function of **`scdiffeq`**. function learns the neural differential equation that describes a gene regulatory network. Subsequently, we can draw insights from this representative dynamical description.

```python
# this function trains and optimizes the neural differential equation
sdq.tl.learn(adata)

# The key inputs to this function follow this ideaology:
# dy/dt = F(x) = odeint(ODE, y0, t)
# ODE=self.net = torch.nn.Sequential(
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
            )
# y0
# t

```

Internally, this function calls a series of functions integral to the method:

#### (4a) Split the data into **`train`**, **`test`**, and **`validation`** sets.

```python
sdq.ut.split(adata)
```
#### (4b) Clonal annotation (if applicable)
```python
sdq.tl.clonality(adata)
```

#### (4c)
```python
sdq.tl.sc_diffeqint(adata)
```

The user need not worry about these functions. All parameters can be passed through **`sdq.tl.learn(adata)`**.

