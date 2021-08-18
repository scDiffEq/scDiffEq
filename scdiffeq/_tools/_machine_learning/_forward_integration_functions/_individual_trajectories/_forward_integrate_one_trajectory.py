
import torchdiffeq

def _forward_integrate_one_trajectory(func, formatted_trajectory):

    """
    Forward integrates using the odeint module from torchdiffeq. 
    
    Parameters:
    -----------
    adata
        anndata._core.anndata.AnnData
    
    func
        Neural_ODE. wraps nn.Sequential.         
    
    formatted_trajectory
        type: dict
    
    Returns:
    --------
    y_predicted, y
    
    Notes:
    ------
    (1) func should be `DiffEq.network` or `self.network`
    
    (2) Eventually should be merged with the SDE. 
    """

    y = formatted_trajectory.y
    y0 = formatted_trajectory.y0
    t = formatted_trajectory.t
    
    y_predicted = torchdiffeq.odeint(func, y0, t)

    return y_predicted, y