# -- import packages: ---------------------------------------------------------
import autodevice
import logging
import pathlib
import torch

# -- set type hints: ----------------------------------------------------------
from typing import Optional, Union

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- mix-in cls: --------------------------------------------------------------
class UtilityIOMixIn(object):
    """MixIn container for various model I/O functions."""

    def to(self, device: torch.device) -> None:
        """Send model to a specific device"""
        self.DiffEq = self.DiffEq.to(device)

    def freeze(self) -> None:
        """Freeze lightning model"""
        self.DiffEq.freeze()

    def load(
        self,
        ckpt_path: Union[str, pathlib.Path],
        freeze: Optional[bool] = True,
        device: Optional[torch.device] = autodevice.AutoDevice(),
    ) -> None:
        """Load a model from a path to a checkpoint."""

        self.__update__(locals())

        self.DiffEq = self.DiffEq.load_from_checkpoint(self._ckpt_path)
        self.DiffEq = self.DiffEq.to(self._device)

        if freeze:
            self.freeze()
