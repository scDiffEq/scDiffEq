
__module_name__ = "_formulate_network_model.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
from collections import OrderedDict
import licorice_font as font
import torch
import torchsde
import torchdiffeq


def _enumerate_activation_functions(return_funcs=True, silent=True):

    """
    Enumerate over available activation functions within the torch.nn module.
    Convenience function.

    Parameters:
    -----------
    return_funcs
        default: False
        type: bool

    silent
        default: True
        type: bool

    Returns:
    --------
    returned_activation_functions
        list of available activation functions.
        Only returned if `return_funcs`==True.

    Notes:
    ------
    (1) Prints output of available activation functions.
    """

    not_activation_funcs = ["Tuple", "torch", "Tensor", "warnings", "Optional"]

    activ_funcs = torch.nn.modules.activation.__dir__()
    if not silent:
        print("Available activation functions:\n")

    returned_activation_functions = []

    for n, func in enumerate(activ_funcs):
        if not func.startswith("__"):
            if func not in not_activation_funcs:
                returned_activation_functions.append(func)
                if n % 3 == 0:
                    if not silent:
                        print("{:<35}".format(func), end="\n")
                else:
                    if not silent:
                        print("{:<35}".format(func), end="\t")

    if return_funcs:
        return returned_activation_functions


def _check_pytorch_activation_function(activation_function):

    """
    Asserts that the passed activation function is valid within torch.nn library.
    Convenience function.

    Parameters:
    -----------
    activation_function

    Returns:
    --------
    None, potential assertion

    Notes:
    ------
    """

    assert activation_function.__class__.__name__ in _enumerate_activation_functions(
        return_funcs=True, silent=True
    ), _enumerate_activation_functions(return_funcs=False, silent=False)


def _construct_hidden_layers(
    neural_net, activation_function, hidden_layers, hidden_nodes, dropout,
):

    neural_net[
        "{}_input".format(str(activation_function).strip("()"))
    ] = activation_function
    for n in range(0, hidden_layers):
        neural_net["hidden_layer_{}".format(n)] = torch.nn.Linear(hidden_nodes, hidden_nodes)
        if dropout:
            neural_net["dropout_{}".format(n)] = torch.nn.Dropout(dropout)
        if n != (hidden_layers - 1):
            neural_net[
                "{}_{}".format(str(activation_function).strip("()"), n)
            ] = activation_function
        else:
            neural_net[
                "{}_output".format(str(activation_function).strip("()"))
            ] = activation_function

    return neural_net


def _construct_flexible_neural_network(
    in_dim=2, out_dim=2, layers=3, nodes=5, activation_function=torch.nn.Tanh(), dropout=0.1
):

    """
    Create a neural network given in/out dimensions and hidden unit dimenions.

    Parameters:
    -----------
    in_dim
        default: 2

    out_dim
        default: 2

    layers
        Number of fully connected torch.nn.Linear() hidden layers.
        default: 2

    nodes
        Number of nodes within a fully connected torch.nn.Linear() hidden layer.
        default: 5

    activation_function
        default: nn.Tanh()
        type: torch.nn.modules.activation.<func>

    Returns:
    --------
    nn.Sequential(neural_network)

    Notes:
    ------
    (1) Additional flexibility may be possible.
    """

    _check_pytorch_activation_function(activation_function)
    neural_net = OrderedDict()    
    neural_net["input_layer"] = torch.nn.Linear(in_dim, nodes)
    neural_net = _construct_hidden_layers(neural_net, activation_function, layers, nodes, dropout)
    neural_net["output_layer"] = torch.nn.Linear(nodes, out_dim)

    return torch.nn.Sequential(neural_net)

class Neural_Differential_Equation(torch.nn.Module):

    """
    General purpose is to recapitulate the Fokker-Planck / Population-Balance Equation wherein
    each term is parameterized by a neural network.

    Parameters:
    -----------
    in_dim
        default: 2
        type: int

    out_dim
        default: 2
        type: int

    layers
        default: 2
        type: int

    nodes
        default: 5
        type: int

    activation_function
        default: torch.nn.Tanh()
        type: torch.nn.modules.activation.<func>
    batch_size
        default: 10
        type: int

    brownian_size
        default: 1
        type: int

    Returns:
    --------
    Instantiates the class: _FokkerPlanck_NeuralSDE
        As described below in Note 2, this is designed to be called as a subclass within a model class.

    Notes:
    ------

    (1) `f` and `g`, which represent the drift and diffusion functions, respectively are required as written
        for the forward integration passed to `torchsde.sdeint`.

    (2) Docstring from the authors of torchsde:

        Base class for all neural network modules.

        Your models should also subclass this class.

        Modules can also contain other Modules, allowing to nest them in
        a tree structure. You can assign the submodules as regular attributes::

            import torch.nn as nn
            import torch.nn.functional as F

            class Model(nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.conv1 = nn.Conv2d(1, 20, 5)
                    self.conv2 = nn.Conv2d(20, 20, 5)

                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    return F.relu(self.conv2(x))

        Submodules assigned in this way will be registered, and will have their
        parameters converted too when you call :meth:`to`, etc.

        :ivar training: Boolean represents whether this module is in training or
                        evaluation mode.
        :vartype training: bool
    
    (3) Independent sizing of the hidden units for the drift and diffusion terms has not yet been implemented.
    """

    noise_type = 'general'
    sde_type = 'ito'
    
    def __init__(
        self,
        diffusion=True,
        in_dim=2,
        out_dim=2,
        layers=2,
        nodes=5,
        activation_function=torch.nn.Tanh(),
        dropout=0.1,
        batch_size=1,
        brownian_size=1,
        noise_type="general",
        sde_type="ito",
    ):

        super().__init__()
        
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.diffusion = diffusion
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers
        self.nodes = nodes
        self.brownian_size = brownian_size
        self.batch_size = batch_size
        
        self.drift_net = _construct_flexible_neural_network(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            layers=self.layers,
            nodes=self.nodes,
            activation_function=activation_function,
            dropout=dropout
        )
            
        if self.diffusion:
            self.diffusion_net = _construct_flexible_neural_network(
                in_dim=self.in_dim,
                out_dim=self.out_dim * brownian_size,
                layers=self.layers,
                nodes=self.nodes,
                activation_function=activation_function,
            )

    def f(self, t, y):

        """

        Notes:
        ------
        (1) Shape of returned: (batch_size, state_size)
        """
        return self.drift_net(y)

    def g(self, t, y):

        """
        Diffusion function

        Notes:
        ------

        """
        if self.diffusion:
            return self.diffusion_net(y).view(
                self.batch_size, self.in_dim, self.brownian_size
               )
        
        
def _choose_integration_function(diffusion):

    """
    If diffusion is not to be included (i.e., diffusion=False), returns odeint. Otherwise, sdeint is returned.

    Parameters:
    -----------
    diffusion
        indicator if diffusion is included
        type: bool

    Returns:
    --------
    integration_function

    Notes:
    ------
    """

    if diffusion:
        return torchsde.sdeint
    else:
        return torchdiffeq.odeint


def _formulate_network_model(
    diffusion=True,
    device="cpu",
    in_dim=2,
    out_dim=2,
    layers=2,
    nodes=5,
    dropout=0.1,
    activation_function=torch.nn.Tanh(),
    batch_size=1,
    brownian_size=1,
    noise_type="general",
    sde_type="ito",
    silent=False,
    **kwargs
):

    """
    Construct (and correspondingly reduce() the drift-(diffusion) equation. Inherently tied to
    this formulation is the choice of integration implementation  (i.e., torchdiffeq.odeint or
    torchsde.sdeint), which is also returned by this function.

    Parameters:
    -----------
    DiffEq
        Neural Differential Equation instantiated by `sdq.scDiffEq()`

    Returns:
    --------
    Network model
    
    integration_function
        

    Notes:
    ------
    (1) Currently,  the easiest way to define the drift-diffusion equation and sub-compositions of
        terms thereof is to insantiate the entire equation and then remove pieces (e.g., diffusion
        term) that are not desired.
        
    (2) If only drift is wanted, diffusion-related sub-classes are removed from the main `sdq.scDiffEq` class.
    """
    
    network_model = Neural_Differential_Equation(diffusion,     
                                                  in_dim=in_dim,
                                                  out_dim=out_dim,
                                                  layers=layers,
                                                  nodes=nodes,
                                                  dropout=dropout,
                                                  activation_function=activation_function,
                                                  batch_size=batch_size,
                                                  brownian_size=brownian_size,
                                                  noise_type=noise_type,
                                                  sde_type=sde_type,
                                                  **kwargs).to(device)
    
    
    if not diffusion:
        try:
            delattr(network_model.__class__, "g")
        except:
            pass
            
    if not silent:
        font.underline("Neural Differential Equation:", ["BOLD", "BLUE"])
        print(network_model)

    integration_function = _choose_integration_function(diffusion)

    return network_model, integration_function
