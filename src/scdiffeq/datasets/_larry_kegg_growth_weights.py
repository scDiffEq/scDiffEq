

# -- import packages: ----------------------------------------------------------
import logging
import os
import pathlib
import torch

# -- import local dependencies: ------------------------------------------------
from ._figshare_downloader import figshare_downloader

# -- set type hints: -----------------------------------------------------------
from typing import Dict, Union

# -- configure logger: ----------------------------------------------------------
logger = logging.getLogger(__name__)

# -- function: ----------------------------------------------------------------
def larry_kegg_growth_weights(
    data_dir: str = os.getcwd(),
    force_download: bool = False,
):
    """
    Download the Larry Kegg Growth Weights dataset.
    """

    fpath = pathlib.Path(data_dir).joinpath("Weinreb2020_growth-all_kegg.pt")

    if not os.path.exists(fpath) or force_download:
        os.makedirs(data_dir, exist_ok=True)
        figshare_downloader(figshare_id=54635780, write_path=fpath)

    return torch.load(fpath, weights_only=False)
