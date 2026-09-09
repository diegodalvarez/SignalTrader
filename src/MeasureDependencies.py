# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 02:59:54 2026

@author: Diego
"""

import dcor
import pandas as pd
from   sklearn.feature_selection import mutual_info_regression

class Dependence:
    
    def __init__(self) -> None: 
        self.random_state = 42
        
    def measure_dependency(
            self,
            df      : pd.DataFrame,
            target  : str,
            features: list) -> pd.DataFrame: 
        
        mi = (mutual_info_regression(
            X            = df[features],
            y            = df[[target]].values,
            random_state = self.random_state))
        
        df_mi = (pd
            .DataFrame(
                data    = mi,
                index   = features,
                columns = ["mi"])
            .reset_index()
            .rename(columns = {"index": "features"}))
        
        df_setup = (df
            .reset_index()
            .melt(id_vars = ["date", target], var_name = "features")
            .drop(columns = ["date"]))
        
        df_pearson = (df_setup
            .groupby("features")
            .agg("corr")
            .loc[lambda x: x.value != 1]
            [["value"]]
            .reset_index()
            [["features", "value"]]
            .rename(columns = {
                "value"   : "pearson",
                "features": "features"}))
        
        df_spearman = (df_setup
            .groupby("features")
            .agg("corr", method = "spearman")
            .loc[lambda x: x.value != 1]
            .reset_index()
            [["features", "value"]]
            .rename(columns = {
                "value" : "spearman",
                "features": "features"}))
        
        df_dcor = (pd
                   .DataFrame({
                       "features": features,
                       "dcor": [
                           dcor.distance_correlation(df[feature], df[target])
                           for feature in features]}))
        
        df_compare = (df_spearman
            .merge(right = df_pearson, how = "inner", on = ["features"])
            .merge(right = df_mi     , how = "inner", on = ["features"])
            .merge(right = df_dcor   , how = "inner", on = ["features"]))
        
        return df_compare
    
import os
if __name__ == "__main__":
    
    src_path  = os.getcwd()
    repo_path = os.path.abspath(os.path.join(src_path, ".."))
    data_path = os.path.join(repo_path, "data", "tmp.parquet")
    
    df_tmp = (pd
            .read_parquet(path = data_path, engine = "pyarrow")
            [["SPY"]]
            .diff()
            .assign(
                ten_ma      = lambda x: x.SPY.ewm(span = 10, adjust = False).mean().shift(),
                thirty_ma   = lambda x: x.SPY.ewm(span = 30, adjust = False).mean().shift(),
                one_hundred = lambda x: x.SPY.ewm(span = 100, adjust = False).mean().shift())
            .dropna())
    
    target   = "SPY"
    features = list(set(df_tmp.columns.to_list()) - set(["SPY"]))
    
    dependency    = Dependence()
    df_dependency = dependency.measure_dependency(df_tmp, target, features)
    
    display(df_dependency)