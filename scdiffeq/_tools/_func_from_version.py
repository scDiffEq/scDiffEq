
import os
from licorice_font import font_format
from ._versions import configure_version
from ._reconstruct_function import reconstruct_function
from ._hyperparams import HyperParams

note = font_format("NOTE", ["BLUE"])
error = font_format("FileNotFoundError", ["RED"])

def func_from_version(
    version=None,
    model_type="NeuralSDE",
    base_dir=".",
    base_path="scDiffEq_model/lightning_logs/version_{}/hparams.yaml",
):

    version, yaml_base_path, versions_available = configure_version(
        version=version, base_dir=base_dir, base_path=base_path
    ).values()
    
    yaml_path = yaml_base_path.format(version)
    if os.path.exists(yaml_path):
        try:
            hparams = HyperParams(yaml_path)
            print(" - [{}] | Reconstructing function from version: {}".format(note, version))
            return reconstruct_function(hparams)
        except:
            raise ValueError("hparams.yaml was found but there was a problem reconstruction the function.")

    fmt_available = [yaml_base_path.format(v) for v in versions_available]
    print(fmt_available)
    
    raise FileNotFoundError(
        "\n\n - [{}] | hparams.yaml path not found: {}\n\nThe following are available:\n\t{}".format(
            error, yaml_path, *fmt_available
        )
    )
