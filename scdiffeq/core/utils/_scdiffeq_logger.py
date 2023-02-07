from datetime import datetime
import os, glob

from ._autoparse_base_class import AutoParseBase

class scDiffEqLogger(AutoParseBase):
    """
    While Lightning uses automatic logging, we need something one step removed from this to take full
    advantage of their setup within the constraints of our model.
    """

    def __init__(self, model_name="scDiffEq_model", working_dir=os.getcwd()):

        self.__parse__(locals(), public=[None])
        self.creation_count = 0
        self._glob_init = glob.glob(self.default_model_outdir + "/version*")

    @property
    def model_name(self):
        return self._model_name

    @property
    def wd(self):
        return self._working_dir

    @property
    def default_model_outdir(self):
        return os.path.join(self.wd, self.model_name)

    def configure_model_outdir(self):
        if not os.path.exists(self.default_model_outdir):
            os.mkdir(self.default_model_outdir)
            line = "".join(
                [
                    "\n ------------- scDiffEq -------------\n\n\n",
                    f" ---- {datetime.now()} ----\n\n",
                ]
            )

            f = open(self.log_path, mode="w")
            f.write(line)
            f.close()

    @property
    def existing_versions(self):
        return self._glob_init

    @property
    def versioned_model_outdir(self):
        return os.path.join(
            self.default_model_outdir,
            "version_{}".format(len(self.existing_versions)),
        )

    @property
    def log_path(self):
        return os.path.join(self.default_model_outdir, "scDiffEq.log")

    def configure_versioned_model_outdir(self):
        if not os.path.exists(self.versioned_model_outdir):
            if not self.creation_count:
                os.mkdir(self.versioned_model_outdir)
                f = open(self.log_path, mode="a")
                v_path = self.versioned_model_outdir
                v = os.path.basename(v_path)
                line = f"\t{v}\t{datetime.now()}\t{v_path}\n"
                f.write(line)
                f.close()
                self.creation_count += 1

        else:
            print(f"Directory: {self.versioned_model_outdir} already exists!")

    def __call__(self):
        self.configure_model_outdir()
        self.configure_versioned_model_outdir()
