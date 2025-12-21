# -- import packages: ---------------------------------------------------------
import lightning
import logging
import pathlib
import pandas as pd

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- mix-in cls: --------------------------------------------------------------
class LoggingMixIn(object):
    def _configure_CSVLogger(self):
        """ """

        #         if hasattr(self, "_load_version"):
        #             version = self._load_version
        #         else:
        #             version = None

        return lightning.pytorch.loggers.CSVLogger(
            save_dir=self._working_dir,
            name=self._name,
            version=None,  # version,
            prefix="",
            flush_logs_every_n_steps=1,
        )

    @property
    def logger(self):
        if not hasattr(self, "_LOGGER"):
            self._csv_logger = self._configure_CSVLogger()
            if hasattr(self, "_logger") and (not self._logger is None):
                self._LOGGER = [self._csv_logger, self._logger]
            else:
                self._LOGGER = [self._csv_logger]

        return self._LOGGER

    @property
    def version(self) -> int:
        return self.logger[0].version

    @property
    def _metrics_path(self):
        return pathlib.Path(self.logger[0].log_dir).joinpath("metrics.csv")

    @property
    def metrics(self):
        return pd.read_csv(self._metrics_path)
