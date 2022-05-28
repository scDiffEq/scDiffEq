import pandas as pd


def _augment_time(
    adata,
    time_key="Time point",
    key_added="t_augmented",
    TimeAugmentDict={2: 0, 4: 0.01, 6: 0.02},
):

    """Augment the original time points."""
    AugmentedTime = pd.DataFrame(
        {
            time_key: list(TimeAugmentDict.keys()),
            key_added: list(TimeAugmentDict.values()),
        }
    )
    try:
        adata.obs[key_added]
    except:
        tmp_obs = pd.merge(adata.obs, AugmentedTime, on=time_key)
        tmp_obs.index = tmp_obs.index.astype(str)
        adata.obs = tmp_obs