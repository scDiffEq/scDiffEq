import torch.nn as nn
from collections import OrderedDict


def _create_linear_nn(n_layers, in_dim, out_dim, n_weights, m_weights):

    """"""

    model = OrderedDict([])
    layer_type = "LinLayer"

    n_layers = 10
    interjoin = "Tanh"
    count = 0
    in_dim = 4
    n_weights = 10
    out_dim = 4
    m_weights = 15

    for layer in range(n_layers + 1):

        layer_name = "_".join([layer_type, str(layer)])
        if layer == 0:
            model[layer_name] = nn.Linear(in_dim, n_weights)
        elif layer == n_layers:
            if layer % 2 == 0:
                model[layer_name] = nn.Linear(m_weights, out_dim)
            else:
                model[layer_name] = nn.Linear(n_weights, out_dim)
        else:
            if layer % 2 == 0:
                model[layer_name] = nn.Linear(m_weights, n_weights)
            else:
                model[layer_name] = nn.Linear(n_weights, m_weights)

        if layer != n_layers:
            interjoin_label = "_".join([interjoin, str(layer)])
            model[interjoin_label] = nn.Tanh()

    return nn.Sequential(model)


def _create_ODE_AnnData(
    adata,
    n_layers=7,
    n_weights=25,
    m_weights=25,
    in_dim=False,
    out_dim=False,
    return_adata=False,
    ODE_name="ODE",
):

    """

    Parameters:
    -----------
    adata
        AnnData

    n_layer
        Number of layers in the neural network.

    n_weights
        Number of weights.

    m_weights
        Number of weights.

    in_dim
        dimensions passed to the neural network.

    out_dim
        dimensions returned from the neural network.

    return_adata
        boolean indicator indicating whether adata should be returned.
        default: False

    ODE_name
        Name of neural network stored in adata.uns.
        default:
        type: str

    Returns:
    --------
    None
        stores AnnData
    """

    if not in_dim:
        in_dim = adata.X.shape[1]
    if not out_dim:
        out_dim = adata.X.shape[1]

    adata.uns[ODE_name] = _create_linear_nn(
        n_layers=n_layers,
        in_dim=in_dim,
        out_dim=out_dim,
        n_weights=n_weights,
        m_weights=m_weights,
    )

    print(
        "Neural Network ('{}') has been constructed as:\n\n{}\n\n...and stored in AnnData as adata.uns['{}']".format(
            ODE_name, adata.uns[ODE_name], ODE_name
        )
    )

    if return_adata:
        return adata
    
    
class ODEFunc(nn.Module):
    def __init__(selfn_layers, in_dim, out_dim, n_weights, m_weights):
        super(ODEFunc, self).__init__()

        self.net = _create_linear_nn(
            n_layers=n_layers,
            in_dim=in_dim,
            out_dim=out_dim,
            n_weights=n_weights,
            m_weights=m_weights,
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net