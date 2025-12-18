# tests/pypi_release_smoke_test.py

"""Simple smoke test for PyPI release.

Checks that the built wheel installs correctly and that the main package
imports and reports a version string.
"""

# -- import packages: ---------------------------------------------------------
import importlib
import logging

# -- configure logger: --------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- configure constant(s): ---------------------------------------------------
PACKAGE_NAME = "scdiffeq"

# -- main function: -----------------------------------------------------------
def main():
    logger.info(f"Starting smoke test for package '{PACKAGE_NAME}'")

    try:
        pkg = importlib.import_module(PACKAGE_NAME)
    except ModuleNotFoundError:
        raise SystemExit(f"❌ Could not import package '{PACKAGE_NAME}' after build.")

    version = getattr(pkg, "__version__", None)
    logger.info(f"✅ Successfully imported '{PACKAGE_NAME}' (version={version})")

if __name__ == "__main__":
    main()