# -- experimental
# -- considering deprecating the more complicated logging
import pathlib
import pandas as pd

class LoggerBridge:
    def __init__(self, DiffEq, idx: int = 0):
        
        """ """
        
        if not len(DiffEq.loggers) < idx + 1:
            self.logger = DiffEq.loggers[idx]

    @property
    def log_dir(self):
        return pathlib.Path(self.logger.log_dir)
    
    @property
    def log_files(self):
        return list(self.log_dir.glob("*"))
        
    @property
    def metrics_csv_path(self):
        matched = [f for f in self.log_files if "metrics" in f.name]
        if matched:
            return  matched[0]
    
    @property
    def log_df(self):
        if not self.metrics_csv_path is None:
            return pd.read_csv(self.metrics_csv_path)
        
    @property
    def version(self):
        return pathlib.Path(self.log_dir).name
    
