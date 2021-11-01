import os

class _ModelOutput:
    def __init__(self, path=False):

        """"""

        if not path:
            self.path = "./scdiffeq/"

        self.results_dir = "results"
        self.predictions_dir = "predictions"
        self.models_dir = "models"
        self.figures_dir = "figures"

        self.results_path = os.path.join(self.path, self.results_dir)
        self.predictions_path = os.path.join(self.path, self.predictions_dir)
        self.models_path = os.path.join(self.path, self.models_dir)
        self.figures_path = os.path.join(self.path, self.figures_dir)

    def save_results(self):

        """"""

        print("Saving to: {}".format(self.results_path))

    def save_predictions(self):

        """"""

        print("Saving to: {}".format(self.predictions_path))

    def save_models(self):

        """"""

        print("Saving to: {}".format(self.models_path))

    def save_figures(self):

        """"""

        print("Saving to: {}".format(self.figures_path))