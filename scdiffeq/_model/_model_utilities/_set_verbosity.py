
__module_name__ = "_set_verbosity.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


def _set_verbosity(verbosity):

    """
    Set verbosity and silence parameters based on an input.

    Parameters:
    -----------
    verbosity
        indicator of verbosity, which obeys the following logic:
          0: Silent
          1: Normal (False)
          2: Verbose
        type: str, int, or bool

    Returns:
    --------
    [verbose, silent]
        list-formatted tuple of bools indicating verbosity then silence.
        type: list(bool, bool)
    """

    if (verbosity is "silent") or (verbosity is 0):
        verbose, silent = False, True

    elif not verbosity or verbosity is 1:
        verbose, silent = False, False

    elif (verbosity is "verbose") or (verbosity is 2):
        verbose, silent = True, False

    return [verbose, silent]