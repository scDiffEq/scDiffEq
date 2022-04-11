
__module_name__ = "_organize_run_info.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import os
import pydk


def _build_run_signature(
    name,
    version,
    nodes,
    layers,
    lr,
    device,
    seed,
):

    """
    Make a dict of signature terms.

    Parameter:
    ----------
    name
        type: str

    version
        type: str

    seed
        type: str

    nodes
        type: str

    layers
        type: str

    device
        type: str

    lr
        type: str

    Returns:
    --------
    signature, SignatureDict

    Notes:
    ------
    """

    SignatureDict = {
        "name": name,
        "version": version,
        "nodes": nodes,
        "layers": layers,
        "LR": lr,
        "device": device,
        "seed": seed,
    }

    signature_components = []

    for [key, value] in SignatureDict.items():
        if key is "":
            signature_components.append("{}{}".format(key, value))

        elif key in ["name", "version"]:
            signature_components.append("{}".format(value))
        elif key is "seed":
            seed_id = str(value)[-4:]
            signature_components.append("{}:{}".format("seed_id", seed_id))
        elif key in ["seed", "device", "LR"]:
            signature_components.append("{}:{}".format(key, value))
        else:
            signature_components.append("{}{}".format(value, key))

    return "_".join(signature_components), SignatureDict


class RunInfo:
    def __init__(self):
        """"""


def _organize_run_info(
    outdir, evaluate_only, run_group, name, version, nodes, layers, lr, device, seed, verbose,
):

    """
    Organize the run info as a class object. Return signature, make outdirs.

    Parameters:
    -----------
    outdir

    run_group

    name

    version

    seed

    nodes

    layers

    device

    lr


    Returns:
    --------
    run_info
        type: python class:  __main__.RunInfo


    Notes:
    ------
    """

    run_signature, SignatureDict = _build_run_signature(
        name,
        version,
        nodes,
        layers,
        lr,
        device,
        seed,
    )

    run_info = RunInfo()

    for key, value in SignatureDict.items():
        pydk.update_class(run_info, key, value)

    if not run_group:
        run_group = "_".join(["scDiffEq", name, version])
        
    if not evaluate_only:
        pydk.update_class(run_info, "run_group", run_group)
        rungroup_outdir = os.path.join(outdir, run_group)
        run_outdir = os.path.join(rungroup_outdir, run_signature)
        pydk.update_class(run_info, "rungroup_outdir", rungroup_outdir)
        pydk.update_class(run_info, "run_outdir", run_outdir)
        pydk.mkdir_flex(rungroup_outdir, verbose=verbose)
        pydk.mkdir_flex(run_outdir, verbose=verbose)
        pydk.mkdir_flex(os.path.join(run_outdir, "img"), verbose=verbose)
        pydk.mkdir_flex(os.path.join(run_outdir, "model"), verbose=verbose)

    return run_info