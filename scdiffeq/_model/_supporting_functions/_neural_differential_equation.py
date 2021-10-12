# _neural_differential_equation.py
__module_name__ = "_neural_differential_equation.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# package imports #
# --------------- #
import torch


# local imports #
# ------------- #
from ._construct_flexible_neural_network import _construct_flexible_neural_network


class _Neural_Differential_Equation(torch.nn.Module):

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

    def __init__(
        self,
        drift=True,
        diffusion=True,
        in_dim=2,
        out_dim=2,
        layers=2,
        nodes=5,
        activation_function=torch.nn.Tanh(),
        batch_size=10,
        brownian_size=1,
        noise_type="general",
        sde_type="ito",
    ):

        super().__init__()

        self.noise_type = noise_type
        self.sde_type = sde_type
        self.drift = drift
        self.diffusion = diffusion
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers
        self.nodes = nodes
        self.brownian_size = brownian_size
        self.batch_size = batch_size

        if self.drift:
            self.drift_net = _construct_flexible_neural_network(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                layers=self.layers,
                nodes=self.nodes,
                activation_function=activation_function,
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
        if self.drift:
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
