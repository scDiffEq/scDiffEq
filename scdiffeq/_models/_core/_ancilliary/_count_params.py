
def count_params(model):

    """There may be a better / native way to do this in pytorch lightning"""

    param_counts = {"total": 0, "trainable": 0, "non-trainable": 0}

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_counts["trainable"] += parameter.numel()
        else:
            param_counts["non-trainable"] += parameter.numel()
        param_counts["total"] += parameter.numel()

    return param_counts