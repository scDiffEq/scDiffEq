### Network model composition

This is the directory containing the components of the network model.

There are two main components outlined here. 


`VAE` and `NeuralDiffEq`. The `NeuralDiffEq` or `NDE` is core to the method. The VAE is an optional dimension reduction step and if used, the `NeuralDiffEq` is passed to the `VAE` and called from inside of it. Otherwise, the method is run with only the `NeuralDiffEq` and the corresponding inputs.