import glob
import os
import pandas as pd

def _extract_params_from_path_signature(path):
    
    filename = os.path.basename(path)

    ParamDict = {}

    for component in filename.split("_"):
        if "id:" in component:
            ParamDict["seed"] = component.split("id:")[1]
        else:
            for param in ["nodes", "layers"]:
                if param in component:
                    ParamDict[param] = component.split(param)[0]
          
    return ParamDict

def _get_run_dict(run_outpaths):
    
    RunDict = {}
    for n, run in enumerate(run_outpaths):
        RunDict[n] = {}

        run_name = os.path.basename(run)
        log_df_path = os.path.join(run, "status.log")

        RunDict[n]['run_params'] = _extract_params_from_path_signature(run)
        RunDict[n]['log_df'] = pd.read_csv(log_df_path, sep='\t')
        
    return RunDict

def _collate_results(results_path):
    
    """collect and organize runs by parameters tested."""
    
    run_outpaths = glob.glob(results_path + "/*")
    
    RunDict = _get_run_dict(run_outpaths)
    
    ParamDictAll = {}
    for n, subdict in enumerate(RunDict.keys()):
        _df = RunDict[subdict]['log_df'].dropna()

        min_loss = _df['total'].min()
        params = RunDict[n]['run_params']
        params['min_loss'] = min_loss
        ParamDictAll[params['seed']] = params

    param_df = pd.DataFrame.from_dict(ParamDictAll).T.reset_index(drop=True)
    param_df['nodes'] = param_df['nodes'].astype(int)
    param_df = param_df.sort_values(['layers','nodes']).reset_index(drop=True)
    
    return param_df