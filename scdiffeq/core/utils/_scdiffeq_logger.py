from datetime import datetime
from pathlib import Path
import os, glob
import ABCParse

from ._info_message import InfoMessage

class scDiffEqLogger(ABCParse.ABCParse):
    """
    While Lightning uses automatic logging, we need something one step removed from this to take full
    advantage of their setup within the constraints of our model.
    """

    def __init__(
        self,
        model_name="scDiffEq_model",
        ckpt_path = None,
        working_dir=os.getcwd(),
    ):

        self.__parse__(locals(), public=[None])
        self._INFO = InfoMessage()
        self.creation_count = 0

    @property
    def _MODEL_NAME(self):
        return self._model_name
    
    @property
    def _WORKING_DIR(self):
        return self._working_dir

    @property
    def _EXISTING_VERSIONS(self):
        return glob.glob(self.PARENT_MODEL_OUTDIR + "/version*")
    
    @property
    def PARENT_MODEL_OUTDIR(self):
        return os.path.join(self._WORKING_DIR, self._MODEL_NAME)
    
    @property
    def LOG_PATH(self):
        return os.path.join(self.PARENT_MODEL_OUTDIR, "scDiffEq.log")

    @property
    def VERSIONED_MODEL_OUTDIR(self):
        if not self._ckpt_path is None:
            if self._PASSED_CKPT_MATCHES_MODEL_PARENT_DIR:
                return self.VERSION_FROM_CKPT
        
        return os.path.join(
            self.PARENT_MODEL_OUTDIR,
            "version_{}".format(len(self._EXISTING_VERSIONS)),
        )
    
    def _configure_parent_model_outdir(self):
        if not os.path.exists(self.PARENT_MODEL_OUTDIR):
            os.mkdir(self.PARENT_MODEL_OUTDIR)
            line = "".join(
                [
                    "\n ------------- scDiffEq -------------\n\n\n",
                    f" ---- {datetime.now()} ----\n\n",
                ]
            )

            f = open(self.LOG_PATH, mode="w")
            f.write(line)
            f.close()
            
    def _configure_versioned_model_outdir(self):
        
        if not os.path.exists(self.VERSIONED_MODEL_OUTDIR):
            if not self.creation_count:
                os.mkdir(self.VERSIONED_MODEL_OUTDIR)
            f = open(self.LOG_PATH, mode="a")
            v_path = self.VERSIONED_MODEL_OUTDIR
            v = os.path.basename(v_path)
            line = f"\t{v}\t{datetime.now()}\t{v_path}\n"
            f.write(line)
            f.close()
            self.creation_count += 1

        elif self._PASSED_CKPT_MATCHES_MODEL_PARENT_DIR:
            if len(self.CKPT_PATH) < 65:
                self._INFO(f"Loading from checkpoint: {self.CKPT_PATH}")
            else:
                self._INFO(f"Loading from checkpoint:\n\t{self.CKPT_PATH}")
        else:
            self._INFO(f"Directory: {self.VERSIONED_MODEL_OUTDIR} already exists!")

    # -- checkpoint loading: ------------------------------------------------
    @property
    def CKPT_PATH(self):
        return Path(self._ckpt_path).absolute().as_posix()
    
    @property
    def VERSION_FROM_CKPT(self):
        return os.path.dirname(self.CKPT_PATH.split("_logs/version_")[0])
    
    @property
    def _PASSED_CKPT_MATCHES_MODEL_PARENT_DIR(self)->bool:
        compare = [self.CKPT_PATH, self.PARENT_MODEL_OUTDIR]
        common = os.path.commonprefix(compare)
        
        return (common == self.PARENT_MODEL_OUTDIR)
    
    def __call__(self):
        self._configure_parent_model_outdir()
        self._configure_versioned_model_outdir()
