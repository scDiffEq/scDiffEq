
import matplotlib.pyplot as plt

def _saveplot(save_dir="./", save_name="plot.png"):

    plot_savename = os.path.join([save_dir, save_name])

    plt.savefig(plot_savename)
    print("Plot saved at:", plot_savename)