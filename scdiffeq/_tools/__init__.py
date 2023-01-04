
__module_name__ = "__init__.py"
__version__ = "0.0.44"
__doc__ = """tools __init__ module. Sub-package of the main scdiffeq API."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# import functions accessed as sdq.tl.<func>: --------------------------------------------
from ._annotate_predict_cells import annotate_predict_cells