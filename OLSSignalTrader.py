# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 09:44:25 2026

@author: Diego
"""

import numpy as np
import pandas as  pd
import statsmodels.api as sm

class OLSSignalTrader:
    
    def __init__(self) -> None: 
        pass
    
    def _single_signal_ols_run_full_sample(
        self, 
        df            : pd.DataFrame,
        endog         : str,
        exog          : str,
        z_score_window: int,
        q_decile      : int,
        vol_target    : float,
        vol_lookback  : int) -> pd.DataFrame: 
        
        df_model = (sm.OLS(
            endog = df[endog],
            exog  = sm.add_constant(df[exog])).
            fit().
            resid.
            to_frame(name = "resid").
            assign(
                lag_resid    = lambda x: x.resid.shift(),
                resid_zscore = lambda x: 
                    (x.resid - x.resid.ewm(span = z_score_window, adjust = False).mean()) / 
                    x.resid.ewm(span = z_score_window, adjust = False).std(),
                    
                    
                lag_resid_zscore  = lambda x: x.resid_zscore.shift(),
                resid_decile      = lambda x: pd.qcut(
                    x      = x.resid, 
                    q      = q_decile, 
                    labels = [i + 1 for i in range(q_decile)]).shift(),
                
                resid_zscore_decile  = lambda x: pd.qcut(
                    x      = x.resid_zscore,
                    q      = q_decile,
                    labels = [i + 1 for i in range(q_decile)]).shift()))
            
            
    
    def single_signal_ols_run(
        self, 
        df            : pd.DataFrame,
        endog         : str,
        exog          : str,
        z_score_window: int   = 30,
        q_decile      : int   = 10,
        vol_target    : float = 0.1,
        vol_lookback  : int   = 100) -> pd.DataFrame:
        
        # full sample returns
        self._single_signal_ols_run_full_sample(
            df, endog, exog, z_score_window, q_decile, vol_target, vol_lookback)
        
        
    


file     = "tmp.parquet"
df_input = (pd.read_parquet(
    path  = file, engine = "pyarrow"))

OLSSignalTrader().single_signal_ols_run(df_input, endog = "spx_rtn", exog = "vix_diff")