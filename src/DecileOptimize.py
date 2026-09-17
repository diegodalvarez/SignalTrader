# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 01:44:29 2026

@author: Diego
"""

import numpy as np
import pandas as pd

from tqdm import tqdm

class Decile:
    
    def __init__(self) -> None: 
        pass
    
    def _insample_optimize(
        self,
        df      : pd.DataFrame, 
        rtn_name: str,
        signal  : str,
        q       : int = 10) -> pd.DataFrame: 
        
        df_decile = (df
                .sort_index()
                .assign(
                    decile     = lambda x: pd.qcut(x = x[signal], q = q, labels = [i + 1 for i in range(q)]),
                    lag_decile = lambda x: x.decile.shift())
                .reset_index())
        
        df_sharpe = (df_decile
                [["lag_decile", rtn_name]]
                .reset_index(drop = True)
                .groupby("lag_decile")
                .agg(lambda x: x.mean() / x.std() * np.sqrt(252)))
        
        df_tmp_decile = (df_sharpe
                .reset_index()
                .loc[lambda x: x.lag_decile.isin([1,2,9,10])]
                .assign(dgroup = lambda x: np.where(x.lag_decile <= 2, "lgroup", "ugroup")))
        
        df_out = (df_tmp_decile
                .drop(columns = ["lag_decile"])
                .groupby("dgroup")
                .agg("prod")
                .assign(signal_scaler = lambda x: np.where(x[rtn_name] > 0, 1, 0))
                .drop(columns = [rtn_name])
                .merge(right = df_tmp_decile, how = "outer", on = ["dgroup"])
                .rename(columns = {rtn_name: "sharpe"})
                .merge(right = df_decile, how = "outer", on = ["lag_decile"]))
    
        return df_out
    
    def in_sample_optimize(
            self,
            df         : pd.DataFrame,
            rtn_name   : str,
            signal     : str,
            signal_name: str,
            date_name  : str = "date",
            q          : int = 10) -> pd.DataFrame: 
        
        cols = df.columns.to_list()
        if date_name in cols: 
            df = df.set_index("date")
        
        df_out = (df
                  .groupby(signal_name)
                  .apply(self._insample_optimize, rtn_name, signal))
        
        return df_out