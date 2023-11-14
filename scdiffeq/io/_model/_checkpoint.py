
import ABCParse
import pathlib
import torch

class Checkpoint(ABCParse.ABCParse):
    def __init__(self, path: pathlib.Path, *args, **kwargs):
        self.__parse__(locals(), public=["path"])

    @property
    def _fname(self):
        return self.path.name.split(".")[0]

    @property
    def epoch(self):
        if self._fname != "last":
            return int(self._fname.split("=")[1].split("-")[0])
        return self._fname

    @property
    def state_dict(self):
        if not hasattr(self, "_ckpt"):
            self._state_dict = torch.load(self.path)  # ["state_dict"]
        return self._state_dict

    def __repr__(self):
        return f"ckpt epoch: {self.epoch}"
