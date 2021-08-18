import vintools
from _load_EMT_simulation import _load_EMT_simulation


class _DataLoader:

    """"""

    def __init__(self):

        """"""

    def load_EMT_simulation(
        self, destination_path="./scdiffeq_data", silent=False, return_data_path=True
    ):

        """500 simulated EMT trajectories."""

        self.EMT_data_path = _load_EMT_simulation(
            destination_path=destination_path,
            silent=silent,
            return_data_path=return_data_path,
        )
        if not silent:
            print(
                "\n{}\n".format(
                    v.ut.format_pystring("Downloaded data:", ["RED", "BOLD"])
                )
            )
            for data_obj in self.EMT_data_path:
                print("\t{}".format(data_obj))
