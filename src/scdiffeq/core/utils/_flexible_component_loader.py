from collections import OrderedDict
import torch_nets
import torch

import ABCParse

# -- lower-level class: -------
class ComponentStateLoader(ABCParse.ABCParse):
    """
    A lower level class, meant to interact with specific model components.
    Used by FlexibleComponentLoader.
    """

    def __init__(self, component, *args, **kwargs):

        self._COMPONENT = component

    @property
    def _EXPECTED_LOAD_MSG(self):
        return "<All keys matched successfully>"

    @property
    def _MODEL_COMPONENT(self) -> OrderedDict:
        return getattr(self._model.DiffEq, self._COMPONENT)

    @property
    def _STATE(self):
        return self._MODEL_COMPONENT.state_dict()

    def _transfer(self, ckpt):

        TransferStateDict = {}

        for key in ckpt["state_dict"].keys():
            if self._COMPONENT in key:
                new_key = key.split(f"{self._COMPONENT}.")[1]
                TransferStateDict[new_key] = ckpt["state_dict"][key]

        return OrderedDict(TransferStateDict)

    @property
    def ckpt(self):
        return torch.load(self._ckpt_path, map_location=self._map_location)

#    def plot_wandb(self):
#        torch_nets.pl.weights_and_biases(self._STATE)

    def __check__(self):
        if not str(self._msg) == self._EXPECTED_LOAD_MSG:
            raise UserWarning("Check keys - there might have been a problem.")

    def __call__(self, model, ckpt_path, map_location=None):

        self.__parse__(locals(), public=[None])

        ckpt_state = self._transfer(self.ckpt)
        self._msg = self._MODEL_COMPONENT.load_state_dict(ckpt_state)

        self.__check__()


# ---- API-facing class: ---------------------------------------------
class FlexibleComponentLoader(ABCParse.ABCParse):
    """
    Load individual component state dicts. for example just VAE or just DiffEq. Or all of it.
    """
    def __init__(self, model):

        self.__parse__(locals())

    def load_encoder_state(self, ckpt_path):

        self._encoder_ckpt_path = ckpt_path

        self.encoder_loader = ComponentStateLoader("Encoder")
        self.encoder_loader(self.model, self._encoder_ckpt_path)

    def load_decoder_state(self, ckpt_path):
        """"""
        self._decoder_ckpt_path = ckpt_path
        self.decoder_loader = ComponentStateLoader("Decoder")
        self.decoder_loader(self.model, self._decoder_ckpt_path)

    def load_DiffEq_state(self, ckpt_path):
        """"""
        self._diffeq_ckpt_path = ckpt_path
        self.diffeq_loader = ComponentStateLoader("DiffEq")
        self.diffeq_loader(self.model, self._diffeq_ckpt_path)

    def load_VAE(self, ckpt_path):
        """"""
        self.load_encoder_state(ckpt_path)
        self.load_decoder_state(ckpt_path)
