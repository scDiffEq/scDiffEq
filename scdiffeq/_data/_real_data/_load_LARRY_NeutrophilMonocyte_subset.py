import os, glob, torch, tqdm
import numpy as np
import vintools as v

def _check_files(path):
    
    """"""    
    return [os.path.join(os.getcwd(), file) for file in os.listdir(path)]

def _report_differences(value, array):

    if value not in array:
        print("\t{} downloaded.".format(value))
        
def _gdown(gdrive_file_id, output_data_dir):
    
    v.ut.mkdir_flex(output_data_dir)
    before_download = _check_files(output_data_dir)
    executable = "gdown https://drive.google.com/uc?id={} --output {}".format(gdrive_file_id, output_data_dir)
    os.system(executable)
    after_download = _check_files(output_data_dir)

    outnull = [_report_differences(value, before_download) for value in after_download]
    
def _download_LARRY_NeutrophilMonocyte_subset(destination=os.path.join(os.getcwd(), "data/LARRY_NM_subset/")):
    
    """"""
    print("Downloading data to: {}\n".format(destination))
    
    gdrive_file_ids= ["1-OzHs09gSOg-4fpvEU3ys0OshL5bHk8e", "1J7LVXv8tzRFIQQtaMgJYmPCvp6PByK3C", "1NBTnnoaVa4ShKP55mfzbg6Dbt7r5AOm_", "1fDoGJIaIE8AYyh37kKFTY_Hll3CdNdnR"]
    
    for file_id in gdrive_file_ids:
        _gdown(file_id, destination)
        
def _load_previously_cached_LARRY_NM_subset_data(cached_data_path):
    
    print("Loading previously cached data into memory...\n")
    files = glob.glob(cached_data_path + "*")
    DataDict = {}
    for n, file in enumerate(tqdm.notebook.tqdm(files)):
        
        if n == 0:
            print("Returning data as a dictionary of torch.Tensors with the following keys.\n")
        
        name = os.path.basename(file).split(".pt")[0]
        print("\t{}".format(name))
        DataDict[name] = torch.load(file)
        
    return DataDict

def _load_LARRY_NeutrophilMonocyte_subset(cache_datadir=os.getcwd()):
    
    """
    Download performatted LARRY dataset. Neutrophil and monocyte lineage subsets only. 
    
    Parameters:
    -----------
    destination
        path where data will be downloaded
    """
    
    cached_data_path = os.path.join(cache_datadir, "data/LARRY_NM_subset/")   
    
    if os.path.exists(cached_data_path):
        DataDict = _load_previously_cached_LARRY_NM_subset_data(cached_data_path)

    else:
        v.ut.mkdir_flex(os.path.dirname(os.path.dirname(cached_data_path)))
        print("Downloading LARRY neutrophil/monocyte data subset...\n")
        print("This will take ~15s depending on internet connection. Download only occurs once.\n")
        _download_LARRY_NeutrophilMonocyte_subset()
        DataDict = _load_previously_cached_LARRY_NM_subset_data(cached_data_path)
    return DataDict