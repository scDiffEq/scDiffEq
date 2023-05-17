import pandas as pd

def sum_norm_df(df):
    return pd.DataFrame({col: df[col] / df.sum(1) for col in df.columns})