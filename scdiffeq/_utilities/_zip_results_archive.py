import numpy as np
import shutil
import os


def _zip_results_archive(outpath, archive_path=False):

    if not archive_path:
        arhive_path = os.path.dirname(outpath)

    parse_dir = outpath.split("/")

    archive_name_start_here = np.argwhere(np.array(parse_dir) == "scdiffeq_outs").item()
    archive_name = "_".join(parse_dir[archive_name_start_here:])
    archive_name = os.path.join(arhive_path, archive_name)

    shutil.make_archive(archive_name, "zip", outpath)
