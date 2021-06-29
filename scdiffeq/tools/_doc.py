# from ._ml_utils import preflight

def train(adata, diffeq="SDE", learning_rate=0.001):

    """
    Use this function to learn a diffeq that describes the data in adata.X. Choose between ODE and SDE to train a model.

    Parameters:
    -----------

    adata
        AnnData object
        (required)

    diffeq
        options: "ODE", "SDE"
        (required)
        default: "SDE"

    learning_rate
        learning rate
        default: 0.001

    Returns:
    --------
    None
        AnnData is updated in place.
    """
