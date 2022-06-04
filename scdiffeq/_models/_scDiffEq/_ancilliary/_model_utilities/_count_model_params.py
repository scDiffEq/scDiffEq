
__module_name__ = "_count_model_params.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


def _count_pytorch_model_params(model, trainable_only=False):

    """
    General function to count pytorhc model parameters.

    Taken from: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """

    if not trainable_only:
        return sum(param.numel() for param in model.parameters())
    else:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _count_model_params(model):

    ParamCount = {}
    
    VAE = model['VAE']
    NDE = model['NeuralDiffEq']

    if not VAE == None:
        ParamCount["VAE"] = {}
        ParamCount["VAE"]["encoder"] = _count_pytorch_model_params(VAE._encoder)
        ParamCount["VAE"]["decoder"] = _count_pytorch_model_params(VAE._decoder)
        ParamCount["VAE"]["total"] = _count_pytorch_model_params(VAE)

    ParamCount["NeuralDiffEq"] = {}
    ParamCount["NeuralDiffEq"]["drift"] = _count_pytorch_model_params(
        NDE._drift_network
    )
    if NDE:
        ParamCount["NeuralDiffEq"]["diffusion"] = _count_pytorch_model_params(
            NDE._diffusion_network
        )
    ParamCount["NeuralDiffEq"]["total"] = _count_pytorch_model_params(NDE)

    return ParamCount