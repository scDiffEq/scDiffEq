
# from .. import _LOGGING

def _read_logs(logging = _LOGGING):
    with open(logging.log_config.log_fpath, "r") as f:
        _logs = f.readlines()
        f.close()
    return _logs

# logs = _read_logs()        
    
