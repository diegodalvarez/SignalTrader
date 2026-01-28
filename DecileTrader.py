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
        lower_level: int = 2,
        upper_level: int = 9) -> pd.DataFrame: 
    
        middle_part = [i for i in range(lower_level + 1, upper_level)]
    
        df_middle = (df_decile.query(
            "decile == @middle_part").
            assign(signal = lambda x: 0))
    
        df_decile_tmp = (df_decile.query(
            "decile != @middle_part").
            assign(decile = lambda x: x.decile.astype(int)).
            assign(decile_tmp = lambda x: np.where(x.decile <= lower_level, "lower_group", "upper_group")))
    
        df_group_scaler = (df_decile_tmp[
            ["decile_tmp", "sharpe"]].
            groupby("decile_tmp").
            agg("prod").
            assign(tmp = lambda x: np.where(x.sharpe > 0, 1, 0))
            [["tmp"]].
            reset_index())
    
        df_decile_outer = (df_decile_tmp.merge(
            right = df_group_scaler, how = "inner", on = ["decile_tmp"]).
            assign(signal = lambda x: x.tmp * x.sharpe).
            drop(columns = ["decile_tmp", "tmp"]))
    
        df_out = (pd.concat([
            df_middle, df_decile_outer]).
            sort_values("decile"))
    
        return df_out
    
    def _optimize_deciles_matched_sharpe_sign(
        self, 
        df_decile  : pd.DataFrame,
        lower_level: int = 2,
        upper_level: int = 9) -> pd.DataFrame: 
    
        middle_part = [i for i in range(lower_level + 1, upper_level)]
    
        df_middle = (df_decile.query(
            "decile == @middle_part").
            assign(signal = lambda x: 0))
    
        df_decile_tmp = (df_decile.query(
            "decile != @middle_part").
            assign(decile = lambda x: x.decile.astype(int)).
            assign(decile_tmp = lambda x: np.where(x.decile <= lower_level, "lower_group", "upper_group")))
    
        df_group_scaler = (df_decile_tmp[
            ["decile_tmp", "sharpe"]].
            groupby("decile_tmp").
            agg("prod").
            assign(tmp = lambda x: np.where(x.sharpe > 0, 1, 0))
            [["tmp"]].
            reset_index())
    
        df_decile_outer = (df_decile_tmp.merge(
            right = df_group_scaler, how = "inner", on = ["decile_tmp"]).
            assign(signal = lambda x: x.tmp * x.sharpe).
            drop(columns = ["decile_tmp", "tmp"]))
    
        df_out = (pd.concat([
            df_middle, df_decile_outer]).
            sort_values("decile"))
    
        return df_out
    
    def get_decile_backtest(
            self,
            df: pd.DataFrame) -> pd.DataFrame: 
        
        df_decile_sharpe = (df.query(
            "group == 'in_sample'")
            [["decile", "signal_name", "sharpe"]].
            drop_duplicates())
        
        df_opt_signal = (df_decile_sharpe.groupby(
            "signal_name").
            apply(self._optimize_deciles_matched_sharpe_sign).
            reset_index(drop = True).
            drop(columns = ["sharpe"]))
        
        df_out = (df.merge(
            right = df_opt_signal, how = "inner", on = ["decile", "signal_name"]).
            assign(signal_rtn = lambda x: np.sign(x.signal) * x.spx_rtn))
        
        return df_out
        
        
file     = "tmp.parquet"
df_input = (pd.read_parquet(
    path  = file, engine = "pyarrow"))

df_input = (df_input.assign(
    signal1 = lambda x: x.vix_diff.ewm(span = 10, adjust = False).mean(),
    siganl2 = lambda x: x.vix_diff.ewm(span = 100, adjust = False).mean()).
    reset_index().
    drop(columns = ["SPY", "VIX", "vix_diff"]).
    melt(id_vars = ["date", "spx_rtn"], var_name = "signal_name", value_name = "signal_val").
    set_index("date"))

df_decile_sharpe = DecileTrader().setup_deciles(df_input, "signal_name", "signal_val", "spx_rtn")

df_decile_ts = pd.read_parquet("decile_ts.parquet")
df_ts        = DecileTrader().get_decile_backtest(df_decile_ts)
df_ts.to_parquet(path = "signal_rtn.parquet")