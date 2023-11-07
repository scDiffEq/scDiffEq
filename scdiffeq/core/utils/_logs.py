from abc import abstractmethod
import glob, os
import pandas as pd

from .. import utils
import ABCParse

# -- Base Class: ------
class Logs(ABCParse.ABCParse):
    def __init__(self, path, stage, version=0):
        self.__parse__(locals(), public=[None])

    @property
    def _AVAILABLE_VERSIONS(self):
        versions = glob.glob(f"{self._path}/{self._stage}*/version_*")
        n_versions = len(versions)
        return [int(os.path.basename(v).split("_")[-1]) for v in versions]

    @property
    def _VERSION(self):
        """Sometimes the passed version isn't found..."""
        if self._version in self._AVAILABLE_VERSIONS:
            return self._version
        elif len(self._AVAILABLE_VERSIONS) > 0:
            return max(self._AVAILABLE_VERSIONS)
        else:
            return "VERSION NOT FOUND"

    @property
    def _BASE_PATH(self):
        return os.path.join(
            self._path, f"{self._stage}_logs", f"version_{self._VERSION}"
        )

    @property
    def _METRICS_PATH(self):
        _metrics_path = os.path.join(self._BASE_PATH, "metrics.csv")
        if os.path.exists(_metrics_path):
            return _metrics_path
#         else:
#             _metrics_path = os.path.join(self._BACKUP_BASE_PATH, "metrics.csv")
#             if os.path.exists(_metrics_path):
#                 return _metrics_path
#             raise FileNotFoundError(_metrics_path)

    @property
    def _HPARAMS_PATH(self):
        return os.path.join(self._BASE_PATH, "hparams.yaml")

    @property
    def _CKPTS_PATH(self):
        return os.path.join(self._BASE_PATH, "checkpoints/*")

    @property
    def METRICS(self):
        return pd.read_csv(self._METRICS_PATH)

    @property
    def HPARAMS(self):
        return sdq.tl.HyperParams(self._HPARAMS_PATH)

    @property
    def CKPTS(self):
        return glob.glob(self._CKPTS_PATH)

    @property
    def METRICS_BY_EPOCH(self):
        return self.METRICS.groupby("epoch")

    @property
    def _EPOCH_STEP_IDX(self):
        return self.METRICS[["epoch", "step"]]

    @property
    def _COLUMNS(self):
        return self.METRICS.columns.tolist()

    def _epoch_sum_mean_filtered_on(self, df, regex):
        _df = df[self._COLUMNS].copy()
        for re in regex:
            _df = _df.filter(regex=re)
        return _df.mean().sum()

    @abstractmethod
    def assemble(self):
        ...
        
    @property
    def _NON_ZERO_COLS(self):
        return self.df.columns[self.df.sum(0) > 0].tolist()

    @property
    def _NULL_COLS(self):
        return self.df.columns[self.df.sum(0) == 0].tolist()
    
    def __call__(self):
        
        self.df = self.assemble()
        return self.df[self._NON_ZERO_COLS]

class TrainLogs(Logs):
    def __init__(self, path, stage="train", version=0):
        super().__init__(path=path, stage=stage, version=version)

    # -- specialized:  everything above is consistent b/t pretrian/train.
    @property
    def TRAINING_SINKHORN(self):
        return (
            self.METRICS_BY_EPOCH.apply(
                self._epoch_sum_mean_filtered_on,
                ["^(?!fate_weighted).*$", "sinkhorn", "training"],
            )
            .to_frame()
            .rename({0: "training_sinkhorn"}, axis=1)
        )

    @property
    def VALIDATION_SINKHORN(self):
        return (
            self.METRICS_BY_EPOCH.apply(
                self._epoch_sum_mean_filtered_on,
                ["^(?!fate_weighted).*$", "sinkhorn", "validation"],
            )
            .to_frame()
            .rename({0: "validation_sinkhorn"}, axis=1)
        )

    @property
    def TRAINING_SINKHORN_FATE_WEIGHTED(self):
        return (
            self.METRICS_BY_EPOCH.apply(
                self._epoch_sum_mean_filtered_on,
                ["fate_weighted", "sinkhorn", "training"],
            )
            .to_frame()
            .rename({0: "training_sinkhorn_fate_weighted"}, axis=1)
        )

    @property
    def VALIDATION_SINKHORN_FATE_WEIGHTED(self):
        return (
            self.METRICS_BY_EPOCH.apply(
                self._epoch_sum_mean_filtered_on,
                ["fate_weighted", "sinkhorn", "validation"],
            )
            .to_frame()
            .rename({0: "validation_sinkhorn_fate_weighted"}, axis=1)
        )

    @property
    def TRAINING_FATE_ACCURACY(self):
        return (
            self.METRICS_BY_EPOCH.apply(
                self._epoch_sum_mean_filtered_on,
                ["fate_acc_score", "training"],
            )
            .to_frame()
            .rename({0: "fate_accuracy_training"}, axis=1)
        )

    @property
    def VALIDATION_FATE_ACCURACY(self):
        return (
            self.METRICS_BY_EPOCH.apply(
                self._epoch_sum_mean_filtered_on,
                ["fate_acc_score", "validation"],
            )
            .to_frame()
            .rename({0: "fate_accuracy_validation"}, axis=1)
        )

    @property
    def TRAINING_KL_DIVERGENCE(self):
        return (
            self.METRICS_BY_EPOCH.apply(
                self._epoch_sum_mean_filtered_on,
                ["kl_div", "training"],
            )
            .to_frame()
            .rename({0: "kl_div_training"}, axis=1)
        )

    @property
    def VALIDATION_KL_DIVERGENCE(self):
        return (
            self.METRICS_BY_EPOCH.apply(
                self._epoch_sum_mean_filtered_on,
                ["kl_div", "validation"],
            )
            .to_frame()
            .rename({0: "kl_div_validation"}, axis=1)
        )

    def assemble(self):
        
        _df = pd.concat(
            [
                self.TRAINING_SINKHORN,
                self.VALIDATION_SINKHORN,
                self.TRAINING_SINKHORN_FATE_WEIGHTED,
                self.VALIDATION_SINKHORN_FATE_WEIGHTED,
                self.TRAINING_KL_DIVERGENCE,
                self.VALIDATION_KL_DIVERGENCE,
            ],
            axis=1,
        )
        _df["total_training"] = _df.filter(regex="training").sum(1)
        _df["total_validation"] = _df.filter(regex="validation").sum(1)

        # -- add accuracy scores after summing: -----
        _df["fate_accuracy_training"] = self.TRAINING_FATE_ACCURACY[
            "fate_accuracy_training"
        ].values
        _df["fate_accuracy_validation"] = self.VALIDATION_FATE_ACCURACY[
            "fate_accuracy_validation"
        ].values

        return _df
    



class PretrainLogs(Logs):
    def __init__(self, path, stage="pretrain", version=0):
        super().__init__(path=path, stage=stage, version=version)

    @property
    def PRETRAIN_RL_MSE(self):
        return (
            self.METRICS_BY_EPOCH.apply(
                self._epoch_sum_mean_filtered_on, ["pretrain_rl_mse"]
            )
            .to_frame()
            .rename({0: "pretrain_rl_mse"}, axis=1)
        )

    def assemble(self):
        """overkill but makes future extension easier"""
        return self.PRETRAIN_RL_MSE
