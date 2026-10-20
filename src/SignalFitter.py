# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:23:00 2026

@author: Diego

Pure computation module for fitting signals to returns via OLS. Nothing in
here reads or writes to disk, DataFrames go in and FitResults / DataFrames
come out

Expected input is a long DataFrame with columns
    <date_col> | <group_col> | <signal_col> | <rtn_col>

date_col, group_col and rtn_col describe the data so they are set once in
__init__ (defaults date, ticker, px_rtn). group_col can be any column to fit
within, e.g. a signal name when one return series is tested against several
signals. signal_col changes between runs so it is passed to each fit, and the
fit records it (with every other setting) in the FitResult it returns

Two steps
    1. fit_*      -> FitResult, a dict-like holder of the fitted objects
                     (no time series of predictions in it)
    2. backtest_* -> long DataFrame time series built from a FitResult

    backtest_fitted  : predicted_t = a + b * signal_t (signal already lagged
                       or lagged with lag), position_t = sign(predicted_t),
                       earns rtn_t
    backtest_residual: same day fit (lag = 0), resid_t = y_t - predicted_t,
                       position_t+1 = -sign(resid_t), earns rtn_t+1

Three styles
    full_sample: params from each group's full history (in-sample)
    train_test : params from train only, applied unchanged to test
    expanding  : params on date t only use data through t, the backtests
                 use params through t-1 for the prediction at t
"""

import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from   dataclasses import dataclass, field
from   scipy.stats import t as t_dist
from   typing import Callable, Dict, List, Tuple, Union

@dataclass(eq = False)
class FitResult:

    '''
    Dictionary-like holder of what a fit produced, keyed by group
        full_sample / train_test: fit_result[group] -> statsmodels results
        expanding               : fit_result[group] -> DataFrame of params by date

    style  : "full_sample" | "train_test" | "expanding"
    config : every setting the fit used (signal_col, y_col, lag, vol, split)
    data   : the prepped long frame the fit used (lagged signal, vol norm rtn)
    objects: {group: fitted object}
    splits : {group: {"train_end", "test_start"}}, train_test only
    skipped: groups with too few obs
    '''

    style  : str
    config : dict
    data   : pd.DataFrame
    objects: dict = field(default_factory = dict)
    splits : dict = field(default_factory = dict)
    skipped: list = field(default_factory = list)

    def __getitem__(self, group: str): return self.objects[group]
    def __contains__(self, group: str) -> bool: return group in self.objects
    def __iter__(self): return iter(self.objects)
    def __len__(self) -> int: return len(self.objects)

    def keys(self): return self.objects.keys()
    def values(self): return self.objects.values()
    def items(self): return self.objects.items()

    def __repr__(self) -> str:

        return "FitResult(style = {}, signal_col = {}, y_col = {}, lag = {}, groups = {}, skipped = {})".format(
            self.style, self.config["signal_col"], self.config["y_col"],
            self.config["lag"], len(self.objects), len(self.skipped))

class SignalFitter:

    def __init__(
            self,
            rtn_col  : str = "px_rtn",
            group_col: str = "ticker",
            date_col : str = "date") -> None:

        self.rtn_col    = rtn_col
        self.group_col  = group_col
        self.date_col   = date_col
        self.ann_factor = np.sqrt(252)

        # const + 1 signal -> need at least 3 obs for df_resid > 0
        self.n_params = 2
        self.min_obs  = self.n_params + 1

    def _lag_signal(
            self,
            df        : pd.DataFrame,
            signal_col: str,
            lag       : Union[int, Dict[str, int]] = 0) -> pd.Series:

        '''
        Lags the signal within each group (never across groups). An int
        applies the same lag to every group, a dict applies a group specific
        lag, groups not in the dict get a lag of 0
        '''

        if isinstance(lag, dict) == False:

            if lag == 0:
                return df[signal_col]

            return df.groupby(self.group_col)[signal_col].shift(lag)

        lag_map = (df
            [self.group_col]
            .map(lag)
            .fillna(0)
            .astype(int))

        signal = df[signal_col].copy()

        # one groupby shift per unique lag value rather than per group
        for lag_val in lag_map.drop_duplicates().to_list():

            if lag_val == 0: continue
            mask         = lag_map == lag_val
            signal[mask] = (df
                .loc[mask]
                .groupby(self.group_col)
                [signal_col]
                .shift(lag_val))

        return signal

    def _prep_data(
            self,
            df        : pd.DataFrame,
            signal_col: str,
            lag       : Union[int, Dict[str, int]] = 0,
            vol_norm  : bool = True,
            vol_window: int  = 100,
            vol_lag   : int  = 1) -> Tuple[pd.DataFrame, str]:

        '''
        Lags the signal and (optionally) volatility normalizes the returns,
        all within group. vol_lag = 1 means the ewm std only uses past
        returns, vol_lag = 0 includes the current return

        Returns the prepped data and the name of the target column
        '''

        df_tmp = (df
            [[self.date_col, self.group_col, signal_col, self.rtn_col]]
            .assign(**{self.date_col: lambda x: pd.to_datetime(x[self.date_col])})
            .sort_values([self.group_col, self.date_col])
            .reset_index(drop = True))

        df_tmp[signal_col] = self._lag_signal(df_tmp, signal_col, lag)

        if vol_norm == True:

            y_col = "vol_norm_rtn"

            # groupby ewm runs in cython, much faster than apply per group
            rtn_vol = (df_tmp
                .groupby(self.group_col)
                [self.rtn_col]
                .ewm(span = vol_window, adjust = False)
                .std()
                .reset_index(level = 0, drop = True)
                .groupby(df_tmp[self.group_col])
                .shift(vol_lag))

            df_tmp = (df_tmp.assign(
                rtn_vol      = rtn_vol,
                vol_norm_rtn = lambda x: (x[self.rtn_col] / x.rtn_vol).replace([np.inf, -np.inf], np.nan)))

        else:
            y_col = self.rtn_col

        df_out = (df_tmp
            .dropna(subset = [signal_col, y_col])
            .reset_index(drop = True))

        return df_out, y_col

    def _get_config(
            self,
            signal_col: str,
            y_col     : str,
            lag       : Union[int, Dict[str, int]],
            vol_norm  : bool,
            vol_window: int,
            vol_lag   : int,
            **kwargs) -> dict:

        config = {
            "signal_col": signal_col,
            "y_col"     : y_col,
            "rtn_col"   : self.rtn_col,
            "group_col" : self.group_col,
            "date_col"  : self.date_col,
            "lag"       : lag,
            "vol_norm"  : vol_norm,
            "vol_window": vol_window,
            "vol_lag"   : vol_lag}

        config.update(kwargs)
        return config

    def _warn_skipped(self, skipped: list, reason: str) -> None:

        if len(skipped) > 0:
            warnings.warn("Skipped ({}): {}".format(reason, skipped), UserWarning, stacklevel = 3)

    def _fit_ols(
            self,
            df        : pd.DataFrame,
            signal_col: str,
            y_col     : str) -> sm.regression.linear_model.RegressionResultsWrapper:

        model = (sm
            .OLS(
                endog = df[y_col],
                exog  = sm.add_constant(df[signal_col], has_constant = "add"))
            .fit())

        return model

    def _get_model_params(self, model: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:

        df_out = (pd.DataFrame({
            "params": model.params,
            "tvalue": model.tvalues,
            "pvalue": model.pvalues}).
            rename_axis("variable").
            reset_index().
            assign(
                nobs     = int(model.nobs),
                rsquared = model.rsquared))

        return df_out

    def _split_ratio(
            self,
            df         : pd.DataFrame,
            train_ratio: float,
            min_train  : int) -> Union[Tuple[pd.DataFrame, pd.DataFrame], None]:

        '''
        Time ordered split of a single group (date index) by ratio, returns
        None if there are too few train or test obs
        '''

        df_tmp  = df.sort_index()
        n_train = int(np.floor(len(df_tmp) * train_ratio))

        if (n_train < min_train) or (len(df_tmp) - n_train < 1):
            return None

        return df_tmp.iloc[:n_train], df_tmp.iloc[n_train:]

    def _split_date(
            self,
            df       : pd.DataFrame,
            cutoff   : pd.Timestamp,
            min_train: int) -> Union[Tuple[pd.DataFrame, pd.DataFrame], None]:

        '''
        Split of a single group (date index) on a date, train <= cutoff < test,
        returns None if there are too few train or test obs
        '''

        df_tmp   = df.sort_index()
        df_train = df_tmp.loc[df_tmp.index <= cutoff]
        df_test  = df_tmp.loc[df_tmp.index > cutoff]

        if (len(df_train) < min_train) or (len(df_test) < 1):
            return None

        return df_train, df_test

    def _fit_splits(
            self,
            df_prep   : pd.DataFrame,
            signal_col: str,
            y_col     : str,
            split_func: Callable) -> Tuple[dict, dict, list]:

        '''
        Fits OLS on the train part of every group, returns the models, the
        split dates and the skipped groups
        '''

        models  = {}
        splits  = {}
        skipped = []

        for group, df_tmp in df_prep.groupby(self.group_col):

            split = split_func(df_tmp.set_index(self.date_col))

            if split is None:
                skipped.append(group)
                continue

            models[group] = self._fit_ols(split[0], signal_col, y_col)
            splits[group] = {
                "train_end" : split[0].index.max(),
                "test_start": split[1].index.min()}

        return models, splits, skipped

    def _expanding_ols(
            self,
            y    : np.ndarray,
            X    : np.ndarray,
            names: List[str]) -> pd.DataFrame:

        '''
        Expanding window OLS solved for every date at once. Uses running sums
        of X'X, X'y, y'y and then a batched inverse so there is no loop over
        dates. Matches looping sm.OLS on an expanding window exactly. Assumes
        the first column of X is the constant (needed for rsquared). Row t
        uses data through t
        '''

        n, k   = X.shape
        nobs   = np.arange(1, n + 1)
        XtX    = np.cumsum(np.einsum("ni,nj->nij", X, X), axis = 0)
        Xty    = np.cumsum(X * y[:, None], axis = 0)
        yty    = np.cumsum(y ** 2)
        sum_y  = np.cumsum(y)

        params   = np.full((n, k), np.nan)
        bse      = np.full((n, k), np.nan)
        tvalues  = np.full((n, k), np.nan)
        pvalues  = np.full((n, k), np.nan)
        rsquared = np.full(n, np.nan)

        # need df_resid > 0 and a non-singular X'X (e.g. flat signal early on)
        diag_prod = np.abs(np.prod(np.diagonal(XtX, axis1 = 1, axis2 = 2), axis = 1))
        valid     = (nobs > k) & (np.abs(np.linalg.det(XtX)) > 1e-10 * diag_prod)
        idx       = np.flatnonzero(valid)

        if len(idx) > 0:

            XtX_inv  = np.linalg.inv(XtX[idx])
            beta     = np.einsum("nij,nj->ni", XtX_inv, Xty[idx])
            ssr      = np.maximum(yty[idx] - np.einsum("ni,ni->n", beta, Xty[idx]), 0)
            df_resid = nobs[idx] - k
            sigma2   = ssr / df_resid
            se       = np.sqrt(sigma2[:, None] * np.diagonal(XtX_inv, axis1 = 1, axis2 = 2))
            sst      = yty[idx] - sum_y[idx] ** 2 / nobs[idx]

            with np.errstate(divide = "ignore", invalid = "ignore"):
                tval = beta / se
                r2   = 1 - ssr / sst

            params[idx]   = beta
            bse[idx]      = se
            tvalues[idx]  = tval
            pvalues[idx]  = 2 * t_dist.sf(np.abs(tval), df_resid[:, None])
            rsquared[idx] = r2

        out_dict = {}
        for i, name in enumerate(names):
            out_dict[name + "_param"]  = params[:, i]
            out_dict[name + "_bse"]    = bse[:, i]
            out_dict[name + "_tvalue"] = tvalues[:, i]
            out_dict[name + "_pvalue"] = pvalues[:, i]

        out_dict["nobs"]     = nobs
        out_dict["rsquared"] = rsquared

        return pd.DataFrame(out_dict)

    def _sharpe(self, x: Union[pd.Series, np.ndarray]) -> float:

        x = np.asarray(x, dtype = float)
        x = x[np.isfinite(x)]

        if len(x) < 2:
            return np.nan

        std = np.std(x, ddof = 1)
        if std == 0:
            return np.nan

        return np.mean(x) / std * self.ann_factor

    def get_oos_stats(
            self,
            y    : Union[pd.Series, np.ndarray],
            y_hat: Union[pd.Series, np.ndarray]) -> pd.Series:

        '''
        Out-of-sample stats for any set of predictions (OLS or ML)

        oos_r2 uses a zero forecast as the benchmark (Gu, Kelly, Xiu), so
        oos_r2 > 0 means the forecast beats predicting a zero return
        '''

        y     = np.asarray(y, dtype = float)
        y_hat = np.asarray(y_hat, dtype = float)
        mask  = np.isfinite(y) & np.isfinite(y_hat)
        y     = y[mask]
        y_hat = y_hat[mask]

        if len(y) == 0:
            return pd.Series({
                "oos_nobs"    : 0,
                "oos_r2"      : np.nan,
                "oos_corr"    : np.nan,
                "oos_hit_rate": np.nan})

        oos_r2   = 1 - np.sum((y - y_hat) ** 2) / np.sum(y ** 2)
        oos_corr = (
            np.corrcoef(y, y_hat)[0, 1]
            if (len(y) > 1) and (np.std(y) > 0) and (np.std(y_hat) > 0)
            else np.nan)

        hit_rate = np.mean(np.sign(y) == np.sign(y_hat))

        out_series = pd.Series({
            "oos_nobs"    : len(y),
            "oos_r2"      : oos_r2,
            "oos_corr"    : oos_corr,
            "oos_hit_rate": hit_rate})

        return out_series

    # Step 1: fits, return FitResult holders

    def fit_full_sample(
            self,
            df        : pd.DataFrame,
            signal_col: str,
            lag       : Union[int, Dict[str, int]] = 0,
            vol_norm  : bool = True,
            vol_window: int  = 100,
            vol_lag   : int  = 1) -> FitResult:

        '''
        Full sample OLS per group, fit_result[group] -> statsmodels results
        '''

        df_prep, y_col = self._prep_data(df, signal_col, lag, vol_norm, vol_window, vol_lag)
        models         = {}
        skipped        = []

        for group, df_tmp in df_prep.groupby(self.group_col):

            if len(df_tmp) < self.min_obs:
                skipped.append(group)
                continue

            models[group] = self._fit_ols(df_tmp.set_index(self.date_col), signal_col, y_col)

        self._warn_skipped(skipped, "too few obs")

        fit_result = FitResult(
            style   = "full_sample",
            config  = self._get_config(signal_col, y_col, lag, vol_norm, vol_window, vol_lag),
            data    = df_prep,
            objects = models,
            skipped = skipped)

        return fit_result

    def fit_train_test_ratio(
            self,
            df         : pd.DataFrame,
            signal_col : str,
            train_ratio: float = 0.7,
            lag        : Union[int, Dict[str, int]] = 0,
            vol_norm   : bool  = True,
            vol_window : int   = 100,
            vol_lag    : int   = 1) -> FitResult:

        '''
        Train / test split by ratio respecting time order, each group is cut
        at its own ratio (first train_ratio of its dates go to train).
        fit_result[group] -> statsmodels results fit on train only
        '''

        if (train_ratio <= 0) or (train_ratio >= 1):
            raise ValueError("train_ratio must be between 0 and 1")

        df_prep, y_col          = self._prep_data(df, signal_col, lag, vol_norm, vol_window, vol_lag)
        models, splits, skipped = self._fit_splits(
            df_prep, signal_col, y_col,
            lambda x: self._split_ratio(x, train_ratio, self.min_obs))

        self._warn_skipped(skipped, "too few train / test obs")

        fit_result = FitResult(
            style   = "train_test",
            config  = self._get_config(
                signal_col, y_col, lag, vol_norm, vol_window, vol_lag,
                split = "ratio", train_ratio = train_ratio),
            data    = df_prep,
            objects = models,
            splits  = splits,
            skipped = skipped)

        return fit_result

    def fit_train_test_date(
            self,
            df         : pd.DataFrame,
            signal_col : str,
            cutoff_date: Union[str, pd.Timestamp],
            lag        : Union[int, Dict[str, int]] = 0,
            vol_norm   : bool = True,
            vol_window : int  = 100,
            vol_lag    : int  = 1) -> FitResult:

        '''
        Train / test split on a common date for every group
        train: date <= cutoff_date, test: date > cutoff_date
        fit_result[group] -> statsmodels results fit on train only
        '''

        cutoff                  = pd.Timestamp(cutoff_date)
        df_prep, y_col          = self._prep_data(df, signal_col, lag, vol_norm, vol_window, vol_lag)
        models, splits, skipped = self._fit_splits(
            df_prep, signal_col, y_col,
            lambda x: self._split_date(x, cutoff, self.min_obs))

        self._warn_skipped(skipped, "too few train / test obs")

        fit_result = FitResult(
            style   = "train_test",
            config  = self._get_config(
                signal_col, y_col, lag, vol_norm, vol_window, vol_lag,
                split = "date", cutoff_date = cutoff),
            data    = df_prep,
            objects = models,
            splits  = splits,
            skipped = skipped)

        return fit_result

    def fit_expanding(
            self,
            df        : pd.DataFrame,
            signal_col: str,
            lag       : Union[int, Dict[str, int]] = 0,
            vol_norm  : bool = True,
            vol_window: int  = 100,
            vol_lag   : int  = 1) -> FitResult:

        '''
        Expanding window OLS per group with no min obs, starts at the first
        date OLS is estimable (3 obs). The row on date t holds params
        estimated with data through t

        fit_result[group] -> DataFrame indexed by date_col with
        const_ / <signal_col>_ param, bse, tvalue, pvalue plus nobs, rsquared
        '''

        df_prep, y_col = self._prep_data(df, signal_col, lag, vol_norm, vol_window, vol_lag)
        names          = ["const", signal_col]
        params         = {}

        for group, df_tmp in df_prep.groupby(self.group_col):

            y = df_tmp[y_col].to_numpy(dtype = float)
            X = np.column_stack([
                np.ones(len(y)),
                df_tmp[signal_col].to_numpy(dtype = float)])

            params[group] = (self
                ._expanding_ols(y, X, names)
                .set_index(pd.Index(df_tmp[self.date_col].values, name = self.date_col)))

        fit_result = FitResult(
            style   = "expanding",
            config  = self._get_config(signal_col, y_col, lag, vol_norm, vol_window, vol_lag),
            data    = df_prep,
            objects = params)

        return fit_result

    def get_params(self, fit_result: FitResult) -> pd.DataFrame:

        '''
        Params table of any FitResult
            full_sample: group | variable | params | tvalue | pvalue | nobs | rsquared
            train_test : same from the train fit plus train_end, test_start
            expanding  : the param history of every group stacked long
        '''

        if fit_result.style == "expanding":

            df_out = (pd.concat([
                df_tmp.assign(**{self.group_col: group})
                for group, df_tmp in fit_result.items()])
                .reset_index())

            return df_out[[self.date_col, self.group_col] + [col for col in df_out.columns if col not in [self.date_col, self.group_col]]]

        df_out = (pd.concat([
            self._get_model_params(model).assign(**{self.group_col: group})
            for group, model in fit_result.items()])
            [[self.group_col, "variable", "params", "tvalue", "pvalue", "nobs", "rsquared"]]
            .reset_index(drop = True))

        if fit_result.style == "train_test":

            df_out = (df_out
                .rename(columns = {"nobs": "train_nobs", "rsquared": "train_rsquared"})
                .assign(
                    train_end  = lambda x: x[self.group_col].map({k: v["train_end"] for k, v in fit_result.splits.items()}),
                    test_start = lambda x: x[self.group_col].map({k: v["test_start"] for k, v in fit_result.splits.items()})))

        return df_out

    # Step 2: backtests, FitResult -> long time series

    def _get_predicted(self, fit_result: FitResult) -> pd.DataFrame:

        '''
        Adds const_param, signal_param, predicted and sample to the prepped
        data of a FitResult
            full_sample: full sample params, sample = in_sample
            train_test : train params on every row, sample = train / test
            expanding  : params through t-1 on row t, sample = expanding
        '''

        if len(fit_result) == 0:
            raise ValueError("FitResult has no fitted groups")

        signal_col = fit_result.config["signal_col"]
        df_data    = (fit_result.data
            .loc[lambda x: x[self.group_col].isin(list(fit_result.keys()))]
            .drop(columns = ["rtn_vol"], errors = "ignore"))

        if fit_result.style == "expanding":

            df_params = (pd.concat([
                df_tmp[["const_param", signal_col + "_param"]].assign(**{self.group_col: group})
                for group, df_tmp in fit_result.items()])
                .rename(columns = {signal_col + "_param": "signal_param"})
                .reset_index())

            df_out = (df_data
                .merge(right = df_params, how = "left", on = [self.date_col, self.group_col])
                .sort_values([self.group_col, self.date_col])
                .reset_index(drop = True)
                .assign(sample = "expanding"))

            # params through t-1 for the prediction on t
            df_out[["const_param", "signal_param"]] = (df_out
                .groupby(self.group_col)
                [["const_param", "signal_param"]]
                .shift(1))

        else:

            df_params = pd.DataFrame({
                self.group_col: list(fit_result.keys()),
                "const_param" : [model.params["const"] for model in fit_result.values()],
                "signal_param": [model.params[signal_col] for model in fit_result.values()]})

            df_out = (df_data
                .merge(right = df_params, how = "inner", on = [self.group_col])
                .sort_values([self.group_col, self.date_col])
                .reset_index(drop = True))

            if fit_result.style == "full_sample":
                df_out = df_out.assign(sample = "in_sample")

            else:
                train_end = {k: v["train_end"] for k, v in fit_result.splits.items()}
                df_out    = df_out.assign(sample = lambda x: np.where(
                    x[self.date_col] <= x[self.group_col].map(train_end), "train", "test"))

        df_out = df_out.assign(predicted = lambda x: x.const_param + x.signal_param * x[signal_col])
        return df_out

    def _size_position(self, x: pd.Series, sizing: str) -> pd.Series:

        if sizing == "sign":
            return np.sign(x)

        if sizing == "raw":
            return x

        raise ValueError("sizing must be one of ['sign', 'raw']")

    def _out_cols(self, fit_result: FitResult, extra: List[str]) -> List[str]:

        y_col = fit_result.config["y_col"]
        cols  = [self.date_col, self.group_col, "sample", fit_result.config["signal_col"], self.rtn_col]

        if y_col != self.rtn_col:
            cols.append(y_col)

        return cols + ["const_param", "signal_param", "predicted"] + extra + ["position", "signal_rtn"]

    def backtest_fitted(
            self,
            fit_result: FitResult,
            sizing    : str  = "sign",
            use_const : bool = True) -> pd.DataFrame:

        '''
        Trades the fitted value, use with a lagged signal (lag >= 1 or a
        signal column that is already lagged)
            predicted_t = const_param + signal_param * signal_t
            position_t  = sign(predicted_t) (sizing = "sign") or predicted_t (sizing = "raw")
            signal_rtn  = position_t * rtn_t (raw returns)

        use_const = False sizes on signal_param * signal_t only, so a large
        constant can't keep the position long all the time

        Returns a long frame, rows without a prediction yet are dropped
        '''

        df_pred  = self._get_predicted(fit_result)
        forecast = df_pred.predicted if use_const else df_pred.signal_param * df_pred[fit_result.config["signal_col"]]

        df_out = (df_pred
            .assign(
                position   = self._size_position(forecast, sizing),
                signal_rtn = lambda x: x.position * x[self.rtn_col])
            .dropna(subset = ["position"])
            [self._out_cols(fit_result, [])]
            .reset_index(drop = True))

        return df_out

    def backtest_residual(
            self,
            fit_result: FitResult,
            sizing    : str = "sign") -> pd.DataFrame:

        '''
        Shorts the same day residual on the next day, needs a fit with lag = 0
            resid_t      = y_t - predicted_t
            position_t+1 = -sign(resid_t) (sizing = "sign") or -resid_t (sizing = "raw")
            signal_rtn   = position_t+1 * rtn_t+1 (raw returns)

        The position is shifted one row within each group. With vol_norm the
        residual is in vol units so sizing = "raw" is close to a z-score

        Returns a long frame, the row on date t holds resid_t and the
        position held on t (from resid_t-1), rows without a position are dropped
        '''

        lag      = fit_result.config["lag"]
        lag_vals = list(lag.values()) if isinstance(lag, dict) else [lag]

        if any(val != 0 for val in lag_vals):
            raise ValueError("backtest_residual needs a same day fit, refit with lag = 0 (got lag = {})".format(lag))

        y_col   = fit_result.config["y_col"]
        df_pred = self._get_predicted(fit_result).assign(resid = lambda x: x[y_col] - x.predicted)

        df_out = (df_pred
            .assign(
                position   = lambda x: -self._size_position(x.resid, sizing).groupby(x[self.group_col]).shift(1),
                signal_rtn = lambda x: x.position * x[self.rtn_col])
            .dropna(subset = ["position"])
            [self._out_cols(fit_result, ["resid"])]
            .reset_index(drop = True))

        return df_out

    def get_backtest_stats(
            self,
            fit_result : FitResult,
            df_backtest: pd.DataFrame) -> pd.DataFrame:

        '''
        Per group and sample: nobs, annualized Sharpe of signal_rtn, hit rate
        (days with a position only), % of days with a position, start, end.

        For backtest_fitted it adds forecast stats of predicted vs y_col
        (r2 against a zero forecast, corr, hit rate). These are only
        out-of-sample on test and expanding rows
        '''

        y_col     = fit_result.config["y_col"]
        is_fitted = "resid" not in df_backtest.columns
        rows      = []

        for (group, sample), df_tmp in df_backtest.groupby([self.group_col, "sample"]):

            active = df_tmp.position != 0
            row    = {
                self.group_col: group,
                "sample"      : sample,
                "nobs"        : len(df_tmp),
                "sharpe"      : self._sharpe(df_tmp.signal_rtn),
                "hit_rate"    : (df_tmp.signal_rtn[active] > 0).mean() if active.sum() > 0 else np.nan,
                "pct_active"  : active.mean(),
                "start"       : df_tmp[self.date_col].min(),
                "end"         : df_tmp[self.date_col].max()}

            if is_fitted:

                stats = self.get_oos_stats(df_tmp[y_col], df_tmp.predicted)
                row.update({
                    "r2"               : stats.oos_r2,
                    "corr"             : stats.oos_corr,
                    "forecast_hit_rate": stats.oos_hit_rate})

            rows.append(row)

        return pd.DataFrame(rows)

# Testing on synthetic data only, no reading or writing

def _simulate_data(
        n_tickers: int   = 5,
        n_obs    : int   = 3_000,
        seed     : int   = 42) -> Tuple[pd.DataFrame, Dict[str, float]]:

    '''
    Simulated panel: persistent AR(1) signal (like an EWMA) and returns that
    load on it with a known beta, tickers start on different dates. The
    signal is already lined up with the return it predicts
    '''

    rng        = np.random.default_rng(seed)
    dates      = pd.bdate_range(start = "2000-01-03", periods = n_obs)
    true_betas = {}
    df_lists   = []

    for i in range(n_tickers):

        ticker = "tick{}_comdty".format(i)
        start  = i * 200
        n      = n_obs - start
        beta   = rng.uniform(0.0005, 0.002) * rng.choice([-1, 1])
        eps    = rng.normal(size = n)
        signal = np.zeros(n)

        for j in range(1, n):
            signal[j] = 0.98 * signal[j - 1] + 0.2 * eps[j]

        px_rtn = 0.0001 + beta * signal + rng.normal(scale = 0.01, size = n)

        df_lists.append(pd.DataFrame({
            "date"  : dates[start:],
            "ticker": ticker,
            "signal": signal,
            "px_rtn": px_rtn}))

        true_betas[ticker] = beta

    # one ticker that only exists late in the sample (tests date split skips)
    df_lists.append(pd.DataFrame({
        "date"  : dates[-50:],
        "ticker": "late_index",
        "signal": rng.normal(size = 50),
        "px_rtn": rng.normal(scale = 0.01, size = 50)}))

    df_out = pd.concat(df_lists).reset_index(drop = True)
    return df_out, true_betas

def _simulate_residual_data(
        n_tickers: int = 4,
        n_obs    : int = 3_000,
        seed     : int = 11) -> pd.DataFrame:

    '''
    Same day loading on the signal plus a residual that mean reverts the next
    day, so shorting yesterday's residual should make money
    '''

    rng      = np.random.default_rng(seed)
    dates    = pd.bdate_range(start = "2000-01-03", periods = n_obs)
    df_lists = []

    for i in range(n_tickers):

        signal = rng.normal(scale = 0.01, size = n_obs)
        eta    = rng.normal(scale = 0.01, size = n_obs)
        resid  = np.zeros(n_obs)

        for j in range(1, n_obs):
            resid[j] = -0.4 * resid[j - 1] + eta[j]

        df_lists.append(pd.DataFrame({
            "date"  : dates,
            "ticker": "tick{}_comdty".format(i),
            "signal": signal,
            "px_rtn": 0.8 * signal + resid}))

    return pd.concat(df_lists).reset_index(drop = True)

def _check(name: str, condition: bool) -> bool:

    print("{:<60} {}".format(name, "PASS" if condition else "FAIL"))
    return bool(condition)

def main() -> None:

    df, true_betas = _simulate_data()
    fitter         = SignalFitter(rtn_col = "px_rtn")
    ticker         = "tick0_comdty"
    cutoff         = "2008-06-30"
    results        = []

    print("Testing SignalFitter on simulated data\n")

    # 1. full sample recovers true betas (raw returns, no vol norm)
    start      = time.time()
    fit_full   = fitter.fit_full_sample(df, "signal", vol_norm = False)
    df_params  = fitter.get_params(fit_full)
    full_time  = time.time() - start

    df_beta = (df_params
        .loc[lambda x: x.variable == "signal"]
        .loc[lambda x: x.ticker.isin(list(true_betas.keys()))]
        .assign(
            true_beta = lambda x: x.ticker.map(true_betas),
            bse       = lambda x: x.params / x.tvalue,
            z_err     = lambda x: np.abs(x.params - x.true_beta) / x.bse))

    results.append(_check("Full sample betas within 4 SE of truth", (df_beta.z_err < 4).all()))

    # 2. FitResult holds the objects and the settings, no time series
    results.append(_check("FitResult: dict-like, fit_result[group] is the model",
        (set(fit_full.keys()) == set(df.ticker)) and hasattr(fit_full[ticker], "params") and (len(fit_full) == df.ticker.nunique())))
    results.append(_check("FitResult: config records signal_col, y_col, lag",
        (fit_full.config["signal_col"] == "signal") and (fit_full.config["y_col"] == "px_rtn") and (fit_full.config["lag"] == 0)))
    results.append(_check("FitResult: no predicted column in the fit data", "predicted" not in fit_full.data.columns))

    # 3. vol_lag = 0 reproduces the notebook's vol_norm_rtn, vol_lag = 1 lags it
    df_prep0, _ = fitter._prep_data(df, "signal", vol_norm = True, vol_window = 100, vol_lag = 0)
    df_nb       = (df
        .loc[lambda x: x.ticker == ticker]
        .set_index("date")
        .assign(vol_norm_rtn = lambda x: x.px_rtn / x.px_rtn.ewm(span = 100, adjust = False).std())
        .dropna())

    df_mine = df_prep0.loc[lambda x: x.ticker == ticker].set_index("date")
    results.append(_check("vol_lag = 0 matches notebook vol_norm_rtn", np.allclose(
        df_nb.vol_norm_rtn.values, df_mine.loc[df_nb.index, "vol_norm_rtn"].values)))

    df_prep1, _ = fitter._prep_data(df, "signal", vol_norm = True, vol_window = 100, vol_lag = 1)
    vol_check   = (df
        .loc[lambda x: x.ticker == ticker]
        .set_index("date")
        .px_rtn
        .ewm(span = 100, adjust = False)
        .std()
        .shift(1))

    df_mine1 = df_prep1.loc[lambda x: x.ticker == ticker].set_index("date")
    results.append(_check("vol_lag = 1 uses lagged ewm std", np.allclose(
        df_mine1.rtn_vol.values, vol_check.loc[df_mine1.index].values)))

    # 4. per group lag does not bleed across groups
    lag_dict  = {"tick0_comdty": 2, "tick1_comdty": 5}
    df_lag, _ = fitter._prep_data(df, "signal", lag = lag_dict, vol_norm = False)
    lag_ok    = True
    for tmp_ticker in df.ticker.drop_duplicates():

        tmp_lag = lag_dict.get(tmp_ticker, 0)
        df_orig = df.loc[lambda x: x.ticker == tmp_ticker].sort_values("date").reset_index(drop = True)
        df_new  = df_lag.loc[lambda x: x.ticker == tmp_ticker].reset_index(drop = True)

        lag_ok  = lag_ok and (len(df_new) == len(df_orig) - tmp_lag)
        lag_ok  = lag_ok and np.allclose(df_new.signal.values, df_orig.signal.values[:len(df_orig) - tmp_lag])
        lag_ok  = lag_ok and (df_new.date.iloc[0] == df_orig.date.iloc[tmp_lag])

    results.append(_check("Per group lag dict shifts within group only", lag_ok))

    # 5. ratio split: time ordered, correct ratio, model fit on train only
    fit_ratio = fitter.fit_train_test_ratio(df, "signal", train_ratio = 0.7)
    ratio_ok  = True
    for group, split in fit_ratio.splits.items():

        df_grp   = fit_ratio.data.loc[lambda x: x.ticker == group]
        n_train  = (df_grp.date <= split["train_end"]).sum()
        ratio_ok = ratio_ok and (split["train_end"] < split["test_start"]) and (n_train == int(np.floor(len(df_grp) * 0.7)))

    results.append(_check("Ratio split: time ordered, correct ratio", ratio_ok))

    df_train = fit_ratio.data.loc[lambda x: (x.ticker == ticker) & (x.date <= fit_ratio.splits[ticker]["train_end"])]
    train_m  = sm.OLS(df_train.vol_norm_rtn.values, sm.add_constant(df_train.signal, has_constant = "add").values).fit()
    results.append(_check("Ratio split: model equals sm.OLS on train rows",
        np.allclose(np.asarray(train_m.params), fit_ratio[ticker].params.values)))

    # 6. date split respects cutoff, skips late group, params ignore test data
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        fit_date = fitter.fit_train_test_date(df, "signal", cutoff_date = cutoff)

    results.append(_check("Date split: train <= cutoff < test", all(
        (split["train_end"] <= pd.Timestamp(cutoff)) and (split["test_start"] > pd.Timestamp(cutoff))
        for split in fit_date.splits.values())))
    results.append(_check("Date split: late group skipped with warning",
        ("late_index" not in fit_date) and ("late_index" in fit_date.skipped) and any("late_index" in str(w.message) for w in caught)))

    df_shock  = df.assign(
        px_rtn = lambda x: np.where(x.date > pd.Timestamp(cutoff), -x.px_rtn * 50, x.px_rtn),
        signal = lambda x: np.where(x.date > pd.Timestamp(cutoff), x.signal * -3, x.signal))
    fit_shock = fitter.fit_train_test_date(df_shock, "signal", cutoff_date = cutoff)
    results.append(_check("Date split: train params ignore test data", all(
        np.allclose(fit_date[k].params.values, fit_shock[k].params.values) for k in fit_date.keys())))

    df_tt_params = fitter.get_params(fit_date)
    results.append(_check("Train / test params table built",
        (len(df_tt_params) == 2 * len(fit_date)) and df_tt_params.pvalue.notna().all() and df_tt_params.train_end.notna().all()))

    # 7. expanding params match looped sm.OLS, no min obs, last row = full sample
    start    = time.time()
    fit_exp  = fitter.fit_expanding(df, "signal")
    exp_time = time.time() - start

    df_exp       = fit_exp[ticker]
    df_exp_data  = fit_exp.data.loc[lambda x: x.ticker == ticker].reset_index(drop = True)
    check_points = [2, 3, 10, 250, 1_000, len(df_exp) - 1]
    exp_ok       = True

    start = time.time()
    for point in check_points:

        df_slice = df_exp_data.iloc[:point + 1]
        model    = sm.OLS(df_slice.vol_norm_rtn, sm.add_constant(df_slice.signal, has_constant = "add")).fit()
        row      = df_exp.iloc[point]
        exp_ok   = exp_ok and np.allclose(
            [row.const_param, row.signal_param, row.const_bse, row.signal_bse, row.signal_pvalue, row.rsquared],
            [model.params.const, model.params.signal, model.bse.const, model.bse.signal, model.pvalues.signal, model.rsquared])

    loop_time = (time.time() - start) / len(check_points)

    results.append(_check("Expanding: params match sm.OLS at sample dates", exp_ok))
    results.append(_check("Expanding: first 2 rows NaN (no min obs)",
        df_exp.signal_param.iloc[:2].isna().all() and df_exp.signal_param.iloc[2:].notna().all()))

    full_vol = fitter.fit_full_sample(df, "signal")[ticker]
    results.append(_check("Expanding: last row equals full sample fit", np.allclose(
        [df_exp.signal_param.iloc[-1], df_exp.signal_tvalue.iloc[-1]],
        [full_vol.params.signal, full_vol.tvalues.signal])))

    # 8. backtest_fitted in-sample: predicted = fittedvalues, position = sign
    start     = time.time()
    bt_full   = fitter.backtest_fitted(fit_full)
    bt_time   = time.time() - start
    bt_tick   = bt_full.loc[lambda x: x.ticker == ticker]

    results.append(_check("Fitted in-sample: predicted equals fittedvalues",
        np.allclose(bt_tick.predicted.values, np.asarray(fit_full[ticker].fittedvalues))))
    results.append(_check("Fitted in-sample: position = sign(predicted), pos * rtn",
        np.array_equal(bt_full.position.values, np.sign(bt_full.predicted.values)) and
        np.allclose(bt_full.signal_rtn.values, (bt_full.position * bt_full.px_rtn).values)))

    bt_noconst = fitter.backtest_fitted(fit_full, use_const = False)
    results.append(_check("Fitted: use_const = False sizes on beta * signal",
        np.array_equal(bt_noconst.position.values, np.sign(bt_noconst.signal_param * bt_noconst.signal).values)))

    # 9. backtest_fitted train / test: train params on test rows, no leakage
    bt_date  = fitter.backtest_fitted(fit_date)
    bt_shock = fitter.backtest_fitted(fit_shock)
    bt_test  = bt_date.loc[lambda x: (x.ticker == ticker) & (x["sample"] == "test")]

    results.append(_check("Fitted train/test: sample labels follow the cutoff",
        (bt_date.loc[lambda x: x["sample"] == "train"].date.max() <= pd.Timestamp(cutoff)) and
        (bt_date.loc[lambda x: x["sample"] == "test"].date.min() > pd.Timestamp(cutoff))))
    results.append(_check("Fitted train/test: test predicted uses train params", np.allclose(
        bt_test.predicted.values,
        fit_date[ticker].params.const + fit_date[ticker].params.signal * bt_test.signal.values)))
    results.append(_check("Fitted train/test: train positions ignore test data", np.array_equal(
        bt_date.loc[lambda x: x["sample"] == "train"].position.values,
        bt_shock.loc[lambda x: x["sample"] == "train"].position.values)))

    # 10. backtest_fitted expanding: params through t-1, future never changes the past
    bt_exp   = fitter.backtest_fitted(fit_exp)
    bt_exp_t = bt_exp.loc[lambda x: x.ticker == ticker].set_index("date")
    point    = 500
    date_t   = df_exp_data.date.iloc[point]
    df_hist  = df_exp_data.iloc[:point]
    model    = sm.OLS(df_hist.vol_norm_rtn, sm.add_constant(df_hist.signal, has_constant = "add")).fit()

    results.append(_check("Fitted expanding: predicted_t uses params through t-1", np.isclose(
        bt_exp_t.loc[date_t, "predicted"],
        model.params.const + model.params.signal * df_exp_data.signal.iloc[point])))

    k        = 1_500
    df_fut   = df.copy()
    late     = df_fut.groupby("ticker").cumcount() > k
    df_fut.loc[late, ["signal", "px_rtn"]] = df_fut.loc[late, ["signal", "px_rtn"]] * -3
    changed  = df_fut.loc[late].groupby("ticker").date.min()

    bt_exp_fut = fitter.backtest_fitted(fitter.fit_expanding(df_fut, "signal"))
    results.append(_check("Fitted expanding: past positions unaffected by future", all(
        np.array_equal(
            bt_exp.loc[lambda x: (x.ticker == t) & (x.date < changed[t])].position.values,
            bt_exp_fut.loc[lambda x: (x.ticker == t) & (x.date < changed[t])].position.values)
        for t in changed.index)))

    df_exp_stats = fitter.get_backtest_stats(fit_exp, bt_exp)
    results.append(_check("Fitted expanding: positive Sharpe on predictive data",
        df_exp_stats.loc[lambda x: x.ticker.isin(list(true_betas.keys()))].sharpe.mean() > 0))

    # 11. backtest_residual: lag guard, resid, next day short, no bleed
    raised = False
    try:
        fitter.backtest_residual(fitter.fit_full_sample(df, "signal", lag = 1))
    except ValueError:
        raised = True

    results.append(_check("Residual: raises on a lagged fit", raised))

    df_rv     = _simulate_residual_data()
    fit_rv    = fitter.fit_full_sample(df_rv, "signal", lag = 0)
    bt_rv     = fitter.backtest_residual(fit_rv)
    bt_rv_t   = bt_rv.loc[lambda x: x.ticker == ticker]
    resid_all = np.asarray(fit_rv[ticker].resid)

    results.append(_check("Residual in-sample: resid equals model resid (one day on)",
        (len(bt_rv_t) == len(resid_all) - 1) and np.allclose(bt_rv_t.resid.values, resid_all[1:])))
    results.append(_check("Residual: position_t = -sign(resid_t-1) within group",
        (len(bt_rv_t) == len(resid_all) - 1) and np.array_equal(bt_rv_t.position.values, -np.sign(resid_all[:-1]))))
    results.append(_check("Residual: one row dropped per group, no bleed",
        (len(bt_rv) == len(fit_rv.data) - len(fit_rv)) and
        (bt_rv.groupby("ticker").date.min() == fit_rv.data.groupby("ticker").date.nth(1).values).all()))

    fit_rv_tt = fitter.fit_train_test_ratio(df_rv, "signal", train_ratio = 0.5)
    bt_rv_tt  = fitter.backtest_residual(fit_rv_tt)
    df_rv_st  = fitter.get_backtest_stats(fit_rv_tt, bt_rv_tt)
    results.append(_check("Residual train/test: positive test Sharpe on reverting data",
        (df_rv_st.loc[lambda x: x["sample"] == "test"].sharpe > 0).all()))

    fit_rv_exp = fitter.fit_expanding(df_rv, "signal")
    bt_rv_exp  = fitter.backtest_residual(fit_rv_exp)
    rv_data    = fit_rv_exp.data.loc[lambda x: x.ticker == ticker].reset_index(drop = True)
    point      = 400
    df_hist    = rv_data.iloc[:point]
    model      = sm.OLS(df_hist.vol_norm_rtn, sm.add_constant(df_hist.signal, has_constant = "add")).fit()
    resid_pt   = rv_data.vol_norm_rtn.iloc[point] - (model.params.const + model.params.signal * rv_data.signal.iloc[point])
    bt_rv_ex_t = bt_rv_exp.loc[lambda x: x.ticker == ticker].set_index("date")

    results.append(_check("Residual expanding: resid_t uses params through t-1",
        np.isclose(bt_rv_ex_t.loc[rv_data.date.iloc[point], "resid"], resid_pt) and
        (bt_rv_ex_t.loc[rv_data.date.iloc[point + 1], "position"] == -np.sign(resid_pt))))

    df_rv_fut = df_rv.copy()
    late_rv   = df_rv_fut.groupby("ticker").cumcount() > k
    df_rv_fut.loc[late_rv, ["signal", "px_rtn"]] = df_rv_fut.loc[late_rv, ["signal", "px_rtn"]] * -3
    changed_rv = df_rv_fut.loc[late_rv].groupby("ticker").date.min()
    bt_rv_fut  = fitter.backtest_residual(fitter.fit_expanding(df_rv_fut, "signal"))

    results.append(_check("Residual expanding: past positions unaffected by future", all(
        np.array_equal(
            bt_rv_exp.loc[lambda x: (x.ticker == t) & (x.date <= changed_rv[t])].position.values,
            bt_rv_fut.loc[lambda x: (x.ticker == t) & (x.date <= changed_rv[t])].position.values)
        for t in changed_rv.index)))

    # 12. custom group_col / date_col give the same results as the defaults
    df_alt     = df.rename(columns = {"ticker": "signal_name", "date": "Date"})
    fitter_alt = SignalFitter(rtn_col = "px_rtn", group_col = "signal_name", date_col = "Date")
    fit_alt    = fitter_alt.fit_train_test_date(df_alt, "signal", cutoff_date = cutoff)
    bt_alt     = fitter_alt.backtest_fitted(fit_alt)

    results.append(_check("Custom group_col / date_col match the defaults",
        ("signal_name" in fitter_alt.get_params(fit_alt).columns) and
        np.allclose(fitter_alt.get_params(fit_alt).params.values, df_tt_params.params.values) and
        np.array_equal(bt_alt.position.values, bt_date.position.values) and
        (list(bt_alt.columns[:2]) == ["Date", "signal_name"])))

    print("\n{} / {} tests passed".format(sum(results), len(results)))
    print("\nTimings")
    print("Full sample fit + params ({} groups)    : {:.3f}s".format(len(fit_full), full_time))
    print("Expanding fit ({} groups, {:,} rows)   : {:.3f}s".format(len(fit_exp), len(fit_exp.data), exp_time))
    print("Looped sm.OLS estimate for same rows    : {:.1f}s".format(loop_time * len(fit_exp.data)))
    print("backtest_fitted in-sample ({:,} rows)  : {:.3f}s".format(len(bt_full), bt_time))

if __name__ == "__main__": main()
