
import os, glob, numpy as np
import ABCParse


class Versions(ABCParse.ABCParse):
    def __init__(
        self, base_dir=".", base_path="scDiffEq_model/lightning_logs/version_{}/hparams.yaml"
    ):

        self.__parse__(locals(), private=["base_dir", "base_path"])

    def __basedir__(self, path):
        return os.path.basename(os.path.dirname(path))

    @property
    def base_path(self):
        return os.path.join(self._base_dir, self._base_path)

    @property
    def available(self):
        return [
            self.__basedir__(version)
            for version in glob.glob(self.base_path.format("*"))
        ]

    @property
    def indices(self):
        return [int(version.split("_")[1]) for version in self.available]

def configure_version(
    version=None,
    base_dir=".",
    base_path="scDiffEq_model/lightning_logs/version_{}/hparams.yaml",
):
    """returns version, yaml_base_path"""

    _versions = Versions(base_dir=base_dir, base_path=base_path)
    base_path = _versions.base_path
    available = _versions.available
    if not version:
        version = _versions.indices[np.argmax(_versions.indices)]
    return {"version": version, "base_path": base_path, "available": available}
