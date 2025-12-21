
# -- import packages: ---------------------------------------------------------
import ABCParse
import pandas as pd


# -- set typing: --------------------------------------------------------------
from typing import Optional


# -- operational class: -------------------------------------------------------
class GroupedMetrics(ABCParse.ABCParse):
    """Container cls for grouping and reporting metric_df by epochs"""

    def __init__(
        self,
        groupby: Optional[str] = "epoch",
        *args,
        **kwargs,
    ) -> None:
        
        """"""
        self.__parse__(locals())

    def _aggr(self, df, regex: str = "training"):
        """ """
        return df.filter(regex=regex).dropna().sum(1).mean()

    @property
    def _GROUPED(self):
        """ """
        return self._metrics_df.groupby(self._groupby)

    def forward(self, stage: str = "training") -> pd.Series:
        """ """
        return self._GROUPED.apply(self._aggr, regex=stage).dropna()

    @property
    def training(self) -> pd.Series:
        """ """
        return self.forward(stage="training")

    @property
    def validation(self) -> pd.Series:
        """ """
        return self.forward(stage="validation")

    @property
    def df(self) -> pd.DataFrame:
        """ """
        df = pd.concat([self.training, self.validation], axis=1).dropna()
        df.columns = ["training", "validation"]
        return df

    @property
    def lr(self):
        return self._GROUPED["opt_param_group_lr"].mean()

    def __call__(self, metrics_df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """

        Args:
            metrics_df (pd.DataFrame)
            
        Returns:
            df (pd.DataFrame): grouped (by epoch) metrics_df.
        """
        self.__update__(locals())

        return self.df
