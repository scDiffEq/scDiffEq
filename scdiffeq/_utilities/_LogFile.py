# import packages #
# --------------- #
import os
import pydk


def _setup_logfile(outdir, columns=["epoch", "d4", "d6", "total", "mode"]):

    status_file = open(os.path.join(outdir, "status.log"), "w")
    header = "\t".join(columns) + "\n"
    status_file.write(header)
    status_file.flush()

    return status_file


def _update_logfile(status_file, epoch, epoch_loss, mode):

    epoch_loss_ = [epoch] + epoch_loss + [sum(epoch_loss)] + [mode]
    epoch_loss_ = [str(item) for item in epoch_loss_]
    status_update = "\t".join(epoch_loss_) + "\n"
    status_file.write(status_update)
    status_file.flush()


class _LogFile:
    def __init__(self, outdir="./scDiffEq_out", columns=["epoch", "d4", "d6", "total", "mode"]):

        self._outdir = outdir
        self._columns = columns
        
        pydk.mkdir_flex(self._outdir)
        pydk.mkdir_flex(os.path.join(self._outdir, "model"))
        self._status_file = _setup_logfile(self._outdir, self._columns)

    def update(self, epoch, epoch_loss, mode):
        
        """
        Notes:
        ------
        (1) Requires previous running of _setup_logfile()
        """

        _update_logfile(self._status_file, epoch, epoch_loss, mode)