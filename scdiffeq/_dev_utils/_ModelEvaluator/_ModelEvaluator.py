
from ._funcs import _get_best_epoch
from ._funcs import _training_summary
from ._funcs import _plot_training_metrics


class ModelEvaluator:
    def __init__(self, model_path, seed=0):

        self._seed = seed
        self._model_path = model_path
        self._summary = _training_summary(self._model_path, self._seed)
        self._best_epoch, self._best_epoch_path = _get_best_epoch(self._summary)
        try:
            self._best_score = float(summary_dict["best_model_score"])
        except:
            pass

    def plot(self, smooth=5):

        _plot_training_metrics(self._model_path, seed=self._seed, smooth=smooth)