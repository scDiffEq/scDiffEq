# -- import packages: ---------------------------------------------------------
import logging
import sys


# -- configure logger: --------------------------------------------------------
def configure_logging(name="scdiffeq", log_file="scdiffeq.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent adding handlers multiple times (important in notebooks!)
    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler (DEBUG+)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    stream_formatter = logging.Formatter("scDiffEq [%(levelname)s]: %(message)s")

    # Stream handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(stream_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
