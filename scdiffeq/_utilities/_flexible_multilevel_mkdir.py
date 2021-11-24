
# _flexible_multilevel_mkdir.py
__module_name__ = "_flexible_multilevel_mkdir.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])

# package imports #
# --------------- #
import os

# local imports #
# ------------- #
from ._format_string_printing_font import _format_string_printing_font


def _flexible_mkdir(path, verbose):

    if os.path.exists(path):
        pass
    elif path == '':
        pass
    else:
        os.mkdir(path)
        if verbose:
            msg = _format_string_printing_font("Directory created", ["BOLD", "CYAN"])
            print("{}: {}".format(msg, path))


def _flexible_multilevel_mkdir(path, verbose=True):

    """
    Create a directory or ignore if already present. Can create multiple levels of directories.
    Parameters:
    -----------
    path
    Returns:
    --------
    None
        os.mkdir(path) (or nothing)
    Notes:
    ------
    (1) supports multi-level directory creation
    """

    parsed_path = []

    for n, directory in enumerate(path.split("/")):
                
        parsed_path.append(directory)
        if len(parsed_path) == (len(path.split("/")) - 1):
            _flexible_mkdir("/".join(parsed_path), verbose)
        else:
            _flexible_mkdir("/".join(parsed_path), verbose=False)