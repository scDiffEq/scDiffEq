# -- import packages: ---------------------------------------------------------
import lightning
import sys

# -- import local dependencies: -----------------------------------------------
from .. import callbacks


# -- config cls: --------------------------------------------------------------
class ProgressBarConfig:
    def __init__(self): ...

    def _detect_env():
        """Returns 'colab', 'jupyter', or 'terminal'."""
        try:
            from IPython import get_ipython

            shell = get_ipython().__class__.__name__
            if "google.colab" in sys.modules:
                return "colab"
            elif shell == "ZMQInteractiveShell":
                return "jupyter"
            else:
                return "terminal"
        except (NameError, ImportError):
            return "terminal"

    def _build_pbar(self, total_epochs: int):

        if self.env == "colab":
            return [lightning.pytorch.callbacks.RichProgressBar()]
        elif self.env == "jupyter":
            return [callbacks.BasicProgressBar(total_epochs=total_epochs)]
        else:
            return []

    @property
    def env(self):
        if not self._env:
            self._env = self._detect_env()
        return self._env

    @property
    def pbar(self):
        if not self._pbar:
            self._pbar = self._build_pbar(total_epochs=self.total_epochs)
        return self._pbar

    @property
    def enable_progress_bar(self):
        if self.env == "jupyter":
            return False
        else:
            return True
