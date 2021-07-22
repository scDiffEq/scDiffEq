

from ._general_utility_functions import _create_random_signature

def _make_spontaneous_figure_dir_figname(scdiffeq_figures_dir = "scdiffeq_figures", extension="pdf"):

    from datetime import date
    import getpass, os
    
    if os.path.exists(scdiffeq_figures_dir):
        pass
    else:
        os.mkdir(scdiffeq_figures_dir)

    user = getpass.getuser()
    today = date.today().isoformat()
    random_signature = _create_random_signature()

    figname = "_".join(["figure", random_signature, user, today]).replace("-", "_")
    figname_with_extension = ".".join([figname, extension])
    fig_outpath = os.path.join(scdiffeq_figures_dir, figname_with_extension)
    
    return fig_outpath

def _make_publication_plot(figsave_path=None, scdiffeq_figures_dir = "scdiffeq_figures", extension="pdf"):
    
    """
    This sets the pdf font type so that it imports correctly in Adobe Illustrator.
    
    Secondary Source: https://gist.github.com/jcheong0428/13a4ec46459a380cd9f7d80ec45acdd6
    Primary Source: http://jonathansoma.com/lede/data-studio/matplotlib/exporting-from-matplotlib-to-open-in-adobe-illustrator/
    """
    
    import matplotlib
    import matplotlib.pyplot as plt
    
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    
    if figsave_path == None:
        figsave_path = _make_spontaneous_figure_dir_figname(scdiffeq_figures_dir=scdiffeq_figures_dir, extension=extension)
    
    # Set base figure size
    plt.savefig(figsave_path)