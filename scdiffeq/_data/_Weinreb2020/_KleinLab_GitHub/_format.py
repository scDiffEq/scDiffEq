
__module_name__ = "_format.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import licorice_font as font
import os
import scipy.io
import scipy.sparse


def _save_as_npz(outpath, file_object, silent=False):

    """
    Save something as .npz
    
    Parameters:
    -----------
    outpath
        path to file.npz
        type: str
    
    file_object
        file to be saved
        type: numpy.ndarray
    
    silent
        default: False
        type: bool
    
    Returns:
    --------
    None
    
    Notes:
    ------
    """

    if not silent:
        print(" - saving to {}...".format(outpath), end=" ")
    scipy.sparse.save_npz(outpath, file_object)
    if not silent:
        print(font.font_format("done.", ["BOLD"]))


def _convert_mtx_to_npz(downloaded_files, outpath, silent):

    """
    Some files are formatted as .mtx, which are slow to read. Here we convert them
    (once) to .npz and then delete the .mtx file.
    
    Parameters:
    -----------
    downloaded_files
        type: dict
    
    outpath
        type: str
    
    silent
        type: bool
    
    Returns:
    --------
    ConvertedDataDict
    
    Notes:
    ------
    
    """

    ConvertedDataDict = {}

    for file, path in downloaded_files.items():
        _outpath = os.path.join(outpath, file + ".npz")
        if os.path.exists(_outpath):
            ConvertedDataDict[file] = _outpath
        else:
            if path.endswith("mtx.gz"):
                if not silent:
                    print(" - reading {}...".format(path), end=" ")
                ConvertedDataDict[file] = scipy.io.mmread(path).tocsc()
                if not silent:
                    print(font.font_format("done.", ["BOLD"]))
                _save_as_npz(_outpath, ConvertedDataDict[file], silent)
            else:
                ConvertedDataDict[file] = False

    return ConvertedDataDict