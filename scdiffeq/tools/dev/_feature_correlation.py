
# -- import packages: ----------------------------------------------------------
import scipy
import anndata
import tqdm
import pandas as pd
import ABCParse


# -- import local dependencies: ------------------------------------------------
from ..core import utils


# -- set typing: ---------------------------------------------------------------
from types import FunctionType


# -- Controller Class: ---------------------------------------------------------
class FeatureCorrelation(ABCParse.ABCParse):
    """
    Calculate the per-cell correlation of a gene feature with a column
    in `adata.obs`.
    """

    def __init__(
        self,
        adata: anndata.AnnData,
        obs_key: str,
        use_key: str = "X_scaled",
        corr_func: FunctionType = scipy.stats.pearsonr,
    ):

        self.__parse__(locals(), public=["adata"])
        self._CorrelationDict = {}

    @property
    def VARS(self):
        return self.adata.var_names

    @property
    def OBS_PROPERTY(self):
        if not hasattr(self, "_obs_property"):
            self._obs_property = self.adata.obs[self._obs_key].values
        return self._obs_property

    @property
    def _PROGRESS_BAR(self):
        return tqdm.notebook.tqdm(
            self.VARS,
            desc = f"Computing gene correlation to {self._obs_key}"
        )

    def _gene(self, gene):
        if self._use_key == "X":
            return self.adata[:, gene].X.flatten()
        else:
            return self.adata[:, gene].layers[self._use_key].flatten()

    def forward(self, gene):
        corr, pval = self._corr_func(self._gene(gene), self.OBS_PROPERTY)
        return {"corr": corr, "pval": pval}

    def _to_frame(self):
        if not hasattr(self, "_corr_df"):
            self._corr_df = pd.DataFrame(self._CorrelationDict).T
        return self._corr_df

    def _annotate_adata(self):
        self.adata.var[f"{self._obs_key}_correlation"] = self._corr_df["corr"].values
        self.adata.var[f"{self._obs_key}_correlation_pval"] = self._corr_df[
            "pval"
        ].values

    def __call__(self, add_to_adata: bool = True):

        for var in self._PROGRESS_BAR:
            self._CorrelationDict[var] = self.forward(var)

        self._to_frame()
        if add_to_adata:
            self._annotate_adata()
        else:
            return self._corr_df


def drift_correlated_features(
    adata: anndata.AnnData,
    drift_key: str = "drift",
    use_key: str = "X_scaled",
    correlation_func: FunctionType = scipy.stats.pearsonr,
    add_to_adata: bool = True,
):
    """
    Compute correlation between per-cell gene values and drift values.
    
    Parameters:
    -----------
    adata
        type: anndata.AnnData
    
    drift_key
        type: str
        default: "drift"
    
    use_key
        type: str
        default: "X_scaled"
        
    correlation_func
        type: FunctionType
        default: scipy.stats.pearsonr
        
        
    add_to_adata
        type: bool
        default: True
        
    
    Returns:
    --------
    None, optionally adata. Updates adata.
    
    Notes:
    ------
    
    """
    
    feature_correlation = FeatureCorrelation(
        adata=adata,
        obs_key=drift_key,
        use_key=use_key,
        corr_func=correlation_func,
    )

    return feature_correlation(add_to_adata=add_to_adata)

def diffusion_correlated_features(
    adata: anndata.AnnData,
    diffusion_key: str = "diffusion",
    use_key: str = "X_scaled",
    correlation_func: FunctionType = scipy.stats.pearsonr,
    add_to_adata=True,
):
    """
    Compute correlation between per-cell gene values and diffusion values.
    
    Parameters:
    -----------
    adata
        type: anndata.AnnData
    
    diffusion_key
        type: str
        default: "diffusion"
    
    use_key
        type: str
        default: "X_scaled"
        
    correlation_func
        type: FunctionType
        default: scipy.stats.pearsonr
        
        
    add_to_adata
        type: bool
        default: True
        
    
    Returns:
    --------
    None, optionally adata. Updates adata.
    
    Notes:
    ------
    
    """
    
    feature_correlation = FeatureCorrelation(
        adata=adata,
        obs_key=diffusion_key,
        use_key=use_key,
        corr_func=correlation_func,
    )

    return feature_correlation(add_to_adata=add_to_adata)

def potential_correlated_features(
    adata: anndata.AnnData,
    potential_key: str = "potential",
    use_key: str = "X_scaled",
    correlation_func: FunctionType = scipy.stats.pearsonr,
    add_to_adata=True,
):
    """
    Compute correlation between per-cell gene values and potential values.
    
    Parameters:
    -----------
    adata
        type: anndata.AnnData
    
    potential_key
        type: str
        default: "psi"
    
    use_key
        type: str
        default: "X_scaled"
        
    correlation_func
        type: FunctionType
        default: scipy.stats.pearsonr
        
        
    add_to_adata
        type: bool
        default: True
        
    
    Returns:
    --------
    None, optionally adata. Updates adata.
    
    Notes:
    ------
    
    """
    
    feature_correlation = FeatureCorrelation(
        adata=adata,
        obs_key=potential_key,
        use_key=use_key,
        corr_func=correlation_func,
    )

    return feature_correlation(add_to_adata=add_to_adata)
