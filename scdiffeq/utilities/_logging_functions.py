
import logging, os

def _initialize_logger(logname="debug.log", _return=False):
    
    """
    Debugging logger.
    
    Parameters:
    -----------
    logname [optional | default: "debug.log"]
        save name. should be unique to avoid overlap and overwrite of previous logs. 
        
    _return [optional | default: False]
    
    
    Returns:
    --------
    logger [optional | default: None]
        Logger object
    """
    import os, logging
    
    logdir="/home/mvinyard/scripts/.process_logs"
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    
    logname = os.path.join(logdir, logname)
    
    if os.path.exists(logname):
        os.remove(logname)
    
    # now we will Create and configure logger
    # set the threshold of logger to INFO 
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s - %(name)s - [ %(message)s ]",
        datefmt="%d-%b-%y | %H:%M:%S",
        force=True,
        handlers=[logging.FileHandler(logname), logging.StreamHandler()],
    )

    # create an object for the logger - this can be loaded elsewhere using `getLogger`
    logger = logging.getLogger()
    
    logger.info("logger created.")
    if _return == True:
        return logger

def _get_logger(
    log_directory="/home/mvinyard/scripts/.process_logs/", log_file="nb.debug.log"
):

    """
    Parameters:
    -----------

    log_directory
        /home/mvinyard/scripts/.process_logs/

    log_file
        nb.debug.log

    Returns:
    --------
    logger
    """

    log_path = os.path.join(log_directory, log_file)
    logger = logging.getLogger(log_path)

    return logger
