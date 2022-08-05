
import torch

def _test_model(model):
    
    """
    assumes model already has the attribute:
        model._dataset['test']
    """

    with torch.no_grad():
        model.func.eval()
        model.evaluate()

    return model.test_loss