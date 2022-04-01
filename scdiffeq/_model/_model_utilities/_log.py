
import os
    
def _setup_logfile(outdir, columns=["epoch", "d2", "d4", "d6", "total", "mode"]):

    status_file = open(os.path.join(outdir, "status.log"), "w")
    header = "\t".join(columns) + "\n"
    status_file.write(header)
    status_file.flush()

    return status_file


def _update_logfile(status_file, epoch, epoch_loss, mode):

    epoch_loss_ = [epoch] + epoch_loss + [sum(epoch_loss)] + [mode]
    epoch_loss_ = [str(loss) for loss in epoch_loss_]
    status_update = "\t".join(epoch_loss_) + "\n"
    status_file.write(status_update)
    status_file.flush()