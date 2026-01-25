# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:03:59 2026

@author: Diego
"""

import numpy as np
import pandas as pd

from tqdm import tqdm

class DecileTrader:
    
    def __init__(self) -> None: 
        pass
    
    def _get_in_sample_decile(
        self, 
        df        : pd.DataFrame,
        signal_val: str,
        q         : int) -> pd.DataFrame: 
        
        df_out = (df.assign(
            decile = lambda x: pd.qcut(
                x      = x[signal_val], 
                q      = q, 
                labels = [i + 1 for i in range(q)]).shift(),
            group = "in_sample"))
        
        return df_out
    
    def _get_out_of_sample_decile(
        self, 
        df        : pd.DataFrame,
        q         : int,
        start_size: int = 30) -> pd.DataFrame: 
        
        dates = df.index.sort_values().to_list()[start_size:]
        dfs   = []
        
        for date in tqdm(dates): 
            
            df_add = (df.query(
                "date <= @date").
                assign(decile = lambda x: pd.qcut(
                    x      = x.signal_val, 
                    q      = q,
                    labels = [i + 1 for i in range(q)]).shift()).
                query("date == date.max()").
                assign(group = "out_of_sample"))
            
            dfs.append(df_add)
            
        df_out = pd.concat(dfs)
        return df_out
        
    def setup_deciles(
        self,
        df         : pd.DataFrame,
        signal_name: str, 
        signal_val : str,
        rtn_name   : str,
        q          : int = 10,
        start_size : int = 30) -> pd.DataFrame: 
        
        
        df_is = (df
                .groupby(signal_name)
                .apply(self._get_in_sample_decile, signal_val, q)
                .drop(columns = [signal_name])
                .reset_index()
                .dropna())
        
        df_os = (df
                .groupby(signal_name)
                .apply(self._get_out_of_sample_decile, q, start_size)
                .drop(columns = [signal_name])
                .reset_index())
        
        df_decile = pd.concat([df_is, df_os])
        
        df_out = (df_decile
                  [["decile", "group", signal_name, rtn_name]]
                  .groupby(["decile", "group", "signal_name"])
                  .agg(lambda x: x.mean() / x.std() * np.sqrt(252))
                  .reset_index()
                  .rename(columns = {rtn_name: "sharpe"})
                  .merge(right = df_decile, how = "inner", on = ["group", "decile", signal_name]))
        
        return df_out
    
    def _optimize_deciles_matched_sharpe_sign(
        self, 
        df_decile  : pd.DataFrame,
        q          : int,
        signal_name: str, 
        lower_level: int = 2,
        upper_level: int = 9) -> pd.DataFrame: 
        
        keep_cols = [
            i for i in range(1, q + 1) if i <= lower_level or i >= upper_level]
        
        df_tmp = (df_decile
                .query("decile == @keep_cols")
                .assign(tmp_name = lambda x: np.where(x.decile <= lower_level, "lower_group", "upper_group")))
        
        df_opt = (df_tmp[
            ["tmp_name", "sharpe"]]
            .groupby("tmp_name")
            .agg("prod")
            .assign(signal = lambda x: np.where(x.sharpe > 0, 1, 0))
            .reset_index()
            .drop(columns = ["sharpe"]))
        
        df_tmp_signal = (df_tmp
                .merge(right = df_opt, how = "inner", on = ["tmp_name"])
                [["decile", "group", signal_name, "signal", "sharpe"]]
                .rename(columns = {"signal": "tmp_signal"})
                .assign(signal = lambda x: np.sign(x.sharpe) * x.tmp_signal)
                .drop(columns = ["tmp_signal"]))
        
        df_middle = (df_decile
                .query("decile != @keep_cols")
                .assign(signal = 0))
        
        df_out = pd.concat([df_tmp_signal, df_middle])
        return df_out
        
    def optimize_deciles_matched_sharpe_sign(
        self,
        df_decile       : pd.DataFrame,
        signal_name     : str, 
        q               : int, 
        lower_level     : int = 2,
        upper_level     : int = 9) -> pd.DataFrame: 
        
        df_opt = (df_decile
                [["decile", "group", signal_name, "sharpe"]]
                .drop_duplicates()
                .groupby(["group", signal_name])
                .apply(self._optimize_deciles_matched_sharpe_sign, q, signal_name, lower_level, upper_level)
                .drop(columns = [signal_name, "group"])
                .reset_index()
                .drop(columns = ["level_2"]))
        
        display(df_opt)
        return-1
        
        
file     = "tmp.parquet"
df_input = (pd.read_parquet(
    path  = file, engine = "pyarrow"))

df_input = (df_input.assign(
    signal1 = lambda x: x.vix_diff.ewm(span = 10, adjust = False).mean(),
    siganl2 = lambda x: x.vix_diff.ewm(span = 100, adjust = False).mean()).
    head(500).
    reset_index().
    drop(columns = ["SPY", "VIX", "vix_diff"]).
    melt(id_vars = ["date", "spx_rtn"], var_name = "signal_name", value_name = "signal_val").
    set_index("date"))

#df_decile_sharpe = DecileTrader().setup_deciles(df_input, "signal_name", "signal_val", "spx_rtn")
#df_decile_sharpe.to_parquet("decile_sharpe.parquet")

df_decile_sharpe = pd.read_parquet(path = "decile_sharpe.parquet")
DecileTrader().optimize_deciles_matched_sharpe_sign(df_decile_sharpe, "signal_name", 10)