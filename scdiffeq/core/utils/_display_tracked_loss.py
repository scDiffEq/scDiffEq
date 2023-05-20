from IPython import display
import os


def display_img(filename):
    display.display(display.Image(filename=filename))
    
def display_tracked_loss(DiffEqLogger):
    fname = os.path.join(
        DiffEqLogger.versioned_model_outdir, "scDiffEq_fit_loss_tracking.png"
    )
    display_img(filename=fname)