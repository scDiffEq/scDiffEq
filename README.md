# Applying neural ODEs to single-cell data and related exploration


So far, I have wrapped my code into a package, which I've been calling "`nodescape`". The main implementation of this package is used as follows:

```
# import packages
import nodesecape as n
import anndata as a    # package for loading single-cell data

# load data
adata = a.read_h5ad("/path/to/adata.h5ad")

# preprocess the data
n.ut.feature_scale(adata)
n.ut.pca(adata)

# split into training / validation / test
data = n.ml.split_test_train(adata)

# define the ODENet function
adata.uns["device"] = n.ml.set_device()
func = adata.uns["odefunc"] = n.ml.ODEFunc(data_dimensionality=2, layers=1, nodes=5).to(n.ml.set_device())

n.ml.train_model(adata)

n.ml.evaluate(adata)
```

The `train_model` function wraps Ricky Chen's original vanilla neural ODE solver, `odeint(func, y0, t)` where one can approximate `y`, in essence using the `odeint` function to solve an initial value problem. For small toy datasets, I run this right in a notebook. For longer jobs, I have written the functions in this <a href="">script</a>a>. This <a href="">notebook</a> shows an example of this implementation of neural ODEs. There are several other, more anciliary functions that help to load / save / process the data as well as several other half-baked functions included in this package. If you want to use this implementation for anything, you can follow the instructions on its <a href="https://github.com/mvinyard/nodescape">README</a> to install it. I'm not particularly attached to it, however - it's moreso just to help me organize it. Open to packaging ideas, etc.

#### A note on single-cell data:
The most popular package for processing single-cell data in python is <a href="https://scanpy.readthedocs.io/en/stable/#">*SCANPY*</a>. There's a data structure that's been popularized around SCANPY called <a href="https://anndata.readthedocs.io/en/stable/">`AnnData`</a>, which is built on `h5py` for efficient storage. I really like it because it's so well organized and useful for storing and sharing datasets and their associated information / metadata.

#### Visualization, dimensional reduction, and quasi-potential landscapes
One of the ways we got started thinking about this project was based conceptually around the developmental landscape. This is a common idealogy used in biology to describe differentiation and development. Thus, many methods have sought to create such a landscape when analyzing single-cell data. Unfortunately, most of these methods just annotate a UMAP or PCA plot or something of the like.

We have been working with some of the developers at NVIDIA who have given us early access to their "*omniverse*" software. I haven't had much time to play with it yet, but they sent us a tarbell of code to use. They generated this cool animation for us a while back.

![landscape_animation](https://github.com/pinellolab/sc-neural-ODEs/bin/landscape_animation.mp4)

#### Notebooks I will clean and upload:
1. Toy simulation of a single hyperparabola attractor from a set of simple, non-linear equations (2-D):
```
def state_eqn(state, t):

    x, y = state[0], state[1]

    xdot = -x + y ** 2
    ydot = -y - x * y

    return np.array([xdot, ydot])
```

![toy-sim-hp-meshgrid](https://i.imgur.com/EVPrSD0.png)
*This particular example uses a mesh grid as `y0` for each trajectory.

2. EMT simulation notebook
    - Working from a set of ordinary differential equations used to describe EMT to building a feature matrix with continuous time included. In most cases, time is stored as `adata.obs['time']`.

3. Training a neural ODE model on 2-D PCA of the EMT simulation

4. Deriving time from velocity (in a simulation)

#### Datasets
1. Toy dataset - based on this <a href="https://scholar.google.com/citations?user=cMBBPisAAAAJ&hl=en#d=gs_md_cita-d&u=%2Fcitations%3Fview_op%3Dview_citation%26hl%3Den%26user%3DcMBBPisAAAAJ%26citation_for_view%3DcMBBPisAAAAJ%3A_FxGoFyzp5QC%26tzom%3D420">paper</a>. This paper is also the foundation of our investigation into this "scLandscape" idea. 
2. EMT simulation - based on this <a href="https://www.pnas.org/content/110/45/18144">paper</a>.
3. <a href="https://www.nature.com/articles/s41588-019-0489-5?proof=t">Real EMT data (CROP-seq)</a>
4. Barcoded LARRY dataset (<a href="https://science.sciencemag.org/content/367/6479/eaaw3381">Weinreb, *Science* 2020</a>)

#### Problems I am facing and questions that need to be answered
1. Training accuracy in spaces of uneven density.
2. What is the best way to infer trajectory in the absence of defined trajectories. In simulations, I have these built-in.

#### Final thoughts on collaboration
- Maybe we can use this GitHub repo to asynchronously update each other on our progress?
- This is my main project so any ideas you have that you feel like you don't have enough bandwidth for, just let me know and we can coordinate.
