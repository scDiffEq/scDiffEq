
__module_name__ = "_download.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import licorice
import os
import wget

def _affix_mtx_npz_suffix(destination, file, download_filename):
    
    """
    Returns either the original download filename or the modified .npz-adjusted path. 
    
    Parameters:
    -----------
    destination
        type: str
    
    file
        type: str
    
    download_filename
        type: str
    
    Returns:
    --------
    download_filename [ potentially npz_path ]
    """
    
    if file.endswith(".mtx.gz"):
        npz_path = os.path.join(destination, file.split(".mtx.gz")[0] + ".npz")
        return npz_path
    else:
        return download_filename
    
def _download_file(base_path, file, DownloadedFiles, name, download_count, verbose):
    
    """
    Construct a url and download a file using wget. 
    
    Parameters:
    -----------
    base_path
    
    file
    
    DownloadedFiles
    
    name
    
    download_count
    
    verbose
    
    Returns:
    --------
    DownloadedFiles
        type: dict
    
    download_count
        type: int
    """

    if download_count == 0 and verbose:
        print("Downloading files from GitHub...\n")

    url = "{}_{}".format(base_path, file)
    DownloadedFiles[name] = wget.download(url)
    if verbose:
        print("\nSaved to: {}\n".format(file))
    download_count += 1

    return DownloadedFiles, download_count


def _download_LARRY_files_from_GitHub(
   base_path, GitHub_repo, file_basenames, destination=os.getcwd(), verbose=True, force=False
):

    """
    Parameters:
    -----------
    destination
        path to download site.
        default: os.getcwd()
        type: str

    verbose
        Optionally recieve more feedback.
        default: True
        type: bool

    force
        Optionally force re-download.
        default: False
        type: bool

    Returns:
    --------
    DownloadedFiles
        Dictionary with structure: {file: filename (with extension)}
        type: dict
    """

    
    DownloadedFiles = {}
    download_count = 0
    for file in file_basenames:
        download_filename = os.path.basename(
            "_".join([base_path, os.path.basename(file)])
        )
        name = file.split(".")[0]
        download_filename = _affix_mtx_npz_suffix(destination, file, download_filename)
        if not os.path.exists(download_filename):
            DownloadedFiles, download_count = _download_file(
                base_path, file, DownloadedFiles, name, download_count, verbose
            )
        elif force:
            DownloadedFiles, download_count = _download_file(
                base_path, file, DownloadedFiles, name, download_count, verbose
            )
        else:
            DownloadedFiles[name] = download_filename
            continue

    if download_count == 0:
        print("All files already downloaded.")
    if verbose:
        message = licorice.font_format("For more information, see:", ["BOLD"])
        print("\n{} {}\n".format(message, GitHub_repo))

    return DownloadedFiles