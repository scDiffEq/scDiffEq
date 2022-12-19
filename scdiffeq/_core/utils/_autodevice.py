
import torch


def autodevice():
    if torch.cuda.is_available():
        return torch.device("cuda:{}".format(torch.cuda.current_device()))
    return torch.device("cpu")