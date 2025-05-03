# -- import packages: ---------------------------------------------------------
import ABCParse
import datetime
import lightning
import time

# -- set type hints: ----------------------------------------------------------
from typing import List, Optional

aliases = {"epoch_train_loss": "train loss", "epoch_validation_loss": "val loss"}


# -- callback cls: ------------------------------------------------------------
class BasicProgressBar(lightning.pytorch.callbacks.Callback):
    def __init__(self, total_epochs, metric_keys: Optional[List[str]] = []) -> None:
        """
        Args:
            total_epochs (int): total number of epochs
            metric_keys (list or None): which metrics to log (e.g., ["val_loss"])
        """
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        self._metric_keys = ["epoch_train_loss", "epoch_validation_loss"]  # metric_keys

    @property
    def metric_keys(self):
        return ABCParse.as_list(self._metric_keys)

    @property
    def _now(self):
        return datetime.datetime.now().strftime("%H:%M:%S")

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        elapsed = time.time() - self.epoch_start_time
        msg = f"[{self._now}] Epoch {current_epoch}/{self.total_epochs} | ({elapsed:.2f}s)"

        # Add metric logs if available
        metrics = trainer.callback_metrics
        metric_parts = []
        for key in self.metric_keys:
            val = metrics.get(key)
            if key in aliases:
                key = aliases[key]
            if val is not None:
                if isinstance(val, (float, int)):
                    metric_parts.append(f"{key}: {val:.2f}")
                else:
                    try:
                        val = float(val)
                        metric_parts.append(f"{key}: {val:.2f}")
                    except:
                        metric_parts.append(f"{key}: {val}")
        if metric_parts:
            msg += " | " + ", ".join(metric_parts)

        print(msg, flush=True)
