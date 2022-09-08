# Experiments

### Comparing the effect of varying drift network size

We try different network sizes of the drift for both the forward_net and potential_net. The `dt` in these experiments is 0.5 (i.e., the model is taking 6 steps). The rest of the parameters can be observed in the linked scripts. For each model, we trained 5 seeds of 100 epochs. This was observed as enough to converge. Each `sigma` (diffusion) network consisted of 2 hidden layers of nodes each.

**2 hidden layers**
400 nodes
* forward_net [model_000.src.py](/src/model_000.src.py)
* potential_net [model_001.src.py](/src/model_001.src.py)

800 nodes
* forward_net [model_002.src.py](/src/model_002.src.py)
* potential_net [model_003.src.py](/src/model_003.src.py)

1600 nodes
* forward_net [model_004.src.py](/src/model_004.src.py)
* potential_net [model_005.src.py](/src/model_005.src.py)

4000 nodes
* forward_net [model_006.src.py](/src/model_006.src.py)
* potential_net [model_007.src.py](/src/model_007.src.py)

**3 hidden layers**
400 nodes
* forward_net [model_008.src.py](/src/model_008.src.py)
* potential_net [model_009.src.py](/src/model_009.src.py)

800 nodes
* forward_net [model_010.src.py](/src/model_010.src.py)
* potential_net [model_011.src.py](/src/model_011.src.py)

1600 nodes
* forward_net [model_012.src.py](/src/model_012.src.py)
* potential_net [model_013.src.py](/src/model_013.src.py)

4000 nodes
* forward_net [model_014.src.py](/src/model_014.src.py)
* potential_net [model_015.src.py](/src/model_015.src.py)

