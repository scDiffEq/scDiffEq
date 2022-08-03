
import os
import pandas as pd
import matplotlib.pyplot as plt
import vinplots


def _load_and_clean_log_csv(path):
    raw_log_df = pd.read_csv(path)
    return raw_log_df.groupby("epoch").mean().dropna().reset_index()

def _get_train_val_loss(df):
    train_loss = df.filter(regex="train").sum(1)
    val_loss = df.filter(regex="val").sum(1)
    return {"train": train_loss, "val":val_loss}

def _plot_model_loss(model=False,
                     log_path=False,
                     colors={"train":"#E3A400", "val":"darkred"},
                     ymax=False,
                    ):
    
    """user can pass a log_path or a model."""
    
    if not log_path:
        log_path = os.path.join(model.logger.log_dir, "metrics.csv")
    
    log_df = _load_and_clean_log_csv(log_path)
    loss_dict = _get_train_val_loss(log_df)
    
    fig = vinplots.Plot()
    fig.construct(nplots=1, ncols=1)
    fig.modify_spines(ax="all", spines_to_delete=['top', 'right'])
    ax = fig.AxesDict[0][0]
    
    for key, df in loss_dict.items():
        ax.scatter(df.index, df.values, label=key, c=colors[key], alpha=0.85)
    
    try:
        name = os.path.basename(model.logger.log_dir)
    except:
        name = os.path.basename(os.path.dirname(log_path))
        
    ax.set_title(name)
    ax.set_xlabel("Epoch")  
    ax.set_ylabel("Wasserstein Distance (epoch mean)")
    ax.legend(edgecolor='w')
    if ymax:
        plt.ylim(0, ymax)
    
    return log_df