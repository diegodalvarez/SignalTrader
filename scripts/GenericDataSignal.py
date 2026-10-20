# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:19:45 2026

@author: Diego
"""

import os
import pandas as pd
import yfinance as yf
import datetime as dt

class GenericDataSignal:
    
    def __init__(self) -> None: 
        
        self.tickers = [
            "SPY", "QQQ", "IYY", "EMB", "VEA", "IEFA", "BND", "IEMG",
            "GLD", "AGG", "IWF", "VWO"]
        
        self.start_date = dt.date(year = 2000, month = 1, day = 1)
        self.end_date   = dt.date(year = 2026, month = 1, day = 1)
        
        self.script_path = os.getcwd()
        self.repo_path   = os.path.abspath(os.path.join(self.script_path, ".."))
        self.data_path   = os.path.join(self.repo_path, "data")
        
    def get_raw_data(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Raw Prices\n")
        out_path = os.path.join(self.data_path, "RawPX.parquet")
        
        if os.path.exists(out_path): 
            if verbose: print("Already have data\n")
            return None
        
        df_raw = (yf
                  .download(
                      tickers     = self.tickers,
                      start       = self.start_date,
                      end         = self.end_date, 
                      auto_adjust = False)
                  .reset_index()
                  .melt(id_vars = [(     'Date',     '')])
                  .rename(columns = {(     'Date',     ''): "Date"}))
        
        if verbose: print("Saving data\n")
        df_raw.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_signal(self, windows: int = [8, 16, 32, 64, 128], verbose: bool = True) -> None: 
        
        if verbose: 
            print("Getting Raw Signal")
            
        out_path = os.path.join(self.data_path, "GenericTrendSignal.parquet")
        if os.path.exists(out_path):
            if verbose: print("Saving data\n")
        
        in_path  = os.path.join(self.data_path, "RawPX.parquet")
        df_setup = (pd
                .read_parquet(path = in_path, engine = "pyarrow")
                .loc[lambda x: x.Price == "Adj Close"]
                .pivot(index = "Date", columns = "Ticker", values = "value"))
        
        df_R = (df_setup
                .diff()
                .apply(lambda x: x / x.ewm(span = 10, adjust = False).std()))
        
        df_lists = []
        
        for window in windows: 
            
            df_signal = (df_R
                    .apply(lambda x: x.ewm(span = window, adjust = False).mean())
                    .reset_index()
                    .melt(id_vars = "Date", value_name = "signal")
                    .dropna())
            
            df_add = (df_signal
                    .set_index("Date")
                    .groupby("Ticker")
                    .apply(lambda x: x.sort_index().signal.shift())
                    .reset_index()
                    .rename(columns = {"signal": "lag_signal"})
                    .merge(right = df_signal, how = "inner", on = ["Date", "Ticker"])
                    .assign(window = window))
            
            df_lists.append(df_add)
            
        df_signals = pd.concat(df_lists)
        df_rtn     = (df_setup
                .pct_change()
                .reset_index()
                .melt(id_vars = "Date", value_name  = "rtn")
                .dropna())
        
        df_out = (df_signals
                .merge(right = df_rtn, how = "inner", on = ["Date", "Ticker"]))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
generic_data = GenericDataSignal()
generic_data.get_raw_data()
generic_data.get_signal()