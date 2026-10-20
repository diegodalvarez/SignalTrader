# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:23:00 2026

@author: Diego

Pure computation module for fitting signals to returns via OLS. Nothing in
here reads or writes to disk, DataFrames go in and dicts / DataFrames come out

Expected input is a long DataFrame with columns
    date | ticker | <signal_col> | <rtn_col>
"""

import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from   scipy.stats import t as t_dist
from   typing import Dict, List, Tuple, Union

class SignalFitter:

    def __init__(
            self,
            signal_col: str = "signal",
            rtn_col   : str = "px_rtn") -> None:

        self.signal_col = signal_col
        self.rtn_col    = rtn_col

        # const + 1 signal -> need at least 3 obs for df_resid > 0
        self.n_params = 2
        self.min_obs  = self.n_params + 1

    def _lag_signal(
            self,
            df : pd.DataFrame,
            lag: Union[int, Dict[str, int]] = 0) -> pd.Series:

        '''
        Lags the signal within each ticker (never across tickers). An int
        applies the same lag to every ticker, a dict applies a ticker specific
        lag, tickers not in the dict get a lag of 0
        '''

        if isinstance(lag, dict) == False:

            if lag == 0:
                return df[self.signal_col]

            return df.groupby("ticker")[self.signal_col].shift(lag)

        lag_map = (df
            .ticker
            .map(lag)
            .fillna(0)
            .astype(int))

        signal = df[self.signal_col].copy()

        # one groupby shift per unique lag value rather than per ticker
        for lag_val in lag_map.drop_duplicates().to_list():

            if lag_val == 0: continue
            mask         = lag_map == lag_val
            signal[mask] = (df
                .loc[mask]
                .groupby("ticker")
                [self.signal_col]
                .shift(lag_val))

        return signal

    def _prep_data(
            self,
            df        : pd.DataFrame,
            lag       : Union[int, Dict[str, int]] = 0,
            vol_norm  : bool = True,
            vol_window: int  = 100,
            vol_lag   : int  = 1) -> Tuple[pd.DataFrame, str]:

        '''
        Lags the signal and (optionally) volatility normalizes the returns,
        all within ticker. vol_lag = 1 means the ewm std only uses past
        returns, vol_lag = 0 includes the current return

        Returns the prepped data and the name of the target column
        '''

        df_tmp = (df
            [["date", "ticker", self.signal_col, self.rtn_col]]
            .assign(date = lambda x: pd.to_datetime(x.date))
            .sort_values(["ticker", "date"])
            .reset_index(drop = True))

        df_tmp[self.signal_col] = self._lag_signal(df_tmp, lag)

        if vol_norm == True:

            y_col = "vol_norm_rtn"

            # groupby ewm runs in cython, much faster than apply per ticker
            rtn_vol = (df_tmp
                .groupby("ticker")
                [self.rtn_col]
                .ewm(span = vol_window, adjust = False)
                .std()
                .reset_index(level = 0, drop = True)
                .groupby(df_tmp.ticker)
                .shift(vol_lag))

            df_tmp = (df_tmp.assign(
                rtn_vol      = rtn_vol,
                vol_norm_rtn = lambda x: (x[self.rtn_col] / x.rtn_vol).replace([np.inf, -np.inf], np.nan)))

        else:
            y_col = self.rtn_col

        df_out = (df_tmp
            .dropna(subset = [self.signal_col, y_col])
            .reset_index(drop = True))

        return df_out, y_col

    def _fit_ols(self, df: pd.DataFrame, y_col: str) -> sm.regression.linear_model.RegressionResultsWrapper:

        model = (sm
            .OLS(
                endog = df[y_col],
                exog  = sm.add_constant(df[self.signal_col], has_constant = "add"))
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
        Time ordered split of a single ticker (date index) by ratio, returns
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
        Split of a single ticker (date index) on a date, train <= cutoff < test,
        returns None if there are too few train or test obs
        '''

        df_tmp   = df.sort_index()
        df_train = df_tmp.loc[df_tmp.index <= cutoff]
        df_test  = df_tmp.loc[df_tmp.index > cutoff]

        if (len(df_train) < min_train) or (len(df_test) < 1):
            return None

        return df_train, df_test

    def _fit_split(
            self,
            df_train: pd.DataFrame,
            df_test : pd.DataFrame,
            y_col   : str) -> dict:

        '''
        Fits on the train set and predicts out-of-sample on the test set
        '''

        model = self._fit_ols(df_train, y_col)

        df_train_out = (df_train.assign(
            predicted = model.fittedvalues))

        df_test_out = (df_test.assign(
            predicted = model.predict(sm.add_constant(df_test[self.signal_col], has_constant = "add"))))

        out_dict = {
            "model": model,
            "train": df_train_out,
            "test" : df_test_out,
            "y_col": y_col}

        return out_dict

    def _expanding_ols(
            self,
            y    : np.ndarray,
            X    : np.ndarray,
            names: List[str]) -> pd.DataFrame:

        '''
        Expanding window OLS solved for every date at once. Uses running sums
        of X'X, X'y, y'y and then a batched inverse so there is no loop over
        dates. Matches looping sm.OLS on an expanding window exactly. Assumes
        the first column of X is the constant (needed for rsquared)
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

        # out-of-sample prediction: params estimated through t-1 applied to X_t
        predicted     = np.full(n, np.nan)
        predicted[1:] = np.einsum("ni,ni->n", params[:-1], X[1:])

        out_dict = {}
        for i, name in enumerate(names):
            out_dict[name + "_param"]  = params[:, i]
            out_dict[name + "_bse"]    = bse[:, i]
            out_dict[name + "_tvalue"] = tvalues[:, i]
            out_dict[name + "_pvalue"] = pvalues[:, i]

        out_dict["nobs"]      = nobs
        out_dict["rsquared"]  = rsquared
        out_dict["predicted"] = predicted

        return pd.DataFrame(out_dict)

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

    def fit_full_sample(
            self,
            df        : pd.DataFrame,
            lag       : Union[int, Dict[str, int]] = 0,
            vol_norm  : bool = True,
            vol_window: int  = 100,
            vol_lag   : int  = 1) -> dict:

        '''
        Full sample OLS per ticker, returns {ticker: statsmodels results}
        '''

        df_prep, y_col = self._prep_data(df, lag, vol_norm, vol_window, vol_lag)
        model_dict     = {}
        skipped        = []

        for ticker, df_tmp in df_prep.groupby("ticker"):

            if len(df_tmp) < self.min_obs:
                skipped.append(ticker)
                continue

            model_dict[ticker] = self._fit_ols(df_tmp.set_index("date"), y_col)

        if len(skipped) > 0:
            warnings.warn("Skipped (too few obs): {}".format(skipped), UserWarning, stacklevel = 2)

        return model_dict

    def get_full_sample_params(self, model_dict: dict) -> pd.DataFrame:

        '''
        Pulls params, tvalues, pvalues, nobs, rsquared out of fit_full_sample
        '''

        df_out = (pd.concat([
            self._get_model_params(model).assign(ticker = ticker)
            for ticker, model in model_dict.items()])
            [["ticker", "variable", "params", "tvalue", "pvalue", "nobs", "rsquared"]].
            reset_index(drop = True))

        return df_out

    def fit_train_test_ratio(
            self,
            df         : pd.DataFrame,
            train_ratio: float = 0.7,
            lag        : Union[int, Dict[str, int]] = 0,
            vol_norm   : bool  = True,
            vol_window : int   = 100,
            vol_lag    : int   = 1) -> dict:

        '''
        Train / test split by ratio respecting time order, each ticker is cut
        at its own ratio (first train_ratio of its dates go to train)

        Returns {ticker: {"model", "train", "test", "y_col"}}
        '''

        if (train_ratio <= 0) or (train_ratio >= 1):
            raise ValueError("train_ratio must be between 0 and 1")

        df_prep, y_col = self._prep_data(df, lag, vol_norm, vol_window, vol_lag)
        tts_dict       = {}
        skipped        = []

        for ticker, df_tmp in df_prep.groupby("ticker"):

            split = self._split_ratio(df_tmp.set_index("date"), train_ratio, self.min_obs)

            if split is None:
                skipped.append(ticker)
                continue

            tts_dict[ticker] = self._fit_split(split[0], split[1], y_col)

        if len(skipped) > 0:
            warnings.warn("Skipped (too few train / test obs): {}".format(skipped), UserWarning, stacklevel = 2)

        return tts_dict

    def fit_train_test_date(
            self,
            df         : pd.DataFrame,
            cutoff_date: Union[str, pd.Timestamp],
            lag        : Union[int, Dict[str, int]] = 0,
            vol_norm   : bool = True,
            vol_window : int  = 100,
            vol_lag    : int  = 1) -> dict:

        '''
        Train / test split on a common date for every ticker
        train: date <= cutoff_date, test: date > cutoff_date

        Returns {ticker: {"model", "train", "test", "y_col"}}
        '''

        cutoff         = pd.Timestamp(cutoff_date)
        df_prep, y_col = self._prep_data(df, lag, vol_norm, vol_window, vol_lag)
        tts_dict       = {}
        skipped        = []

        for ticker, df_tmp in df_prep.groupby("ticker"):

            split = self._split_date(df_tmp.set_index("date"), cutoff, self.min_obs)

            if split is None:
                skipped.append(ticker)
                continue

            tts_dict[ticker] = self._fit_split(split[0], split[1], y_col)

        if len(skipped) > 0:
            warnings.warn("Skipped (too few train / test obs): {}".format(skipped), UserWarning, stacklevel = 2)

        return tts_dict

    def get_train_test_params(self, tts_dict: dict) -> pd.DataFrame:

        '''
        Works on output of fit_train_test_ratio or fit_train_test_date.
        Train params with tvalues / pvalues plus out-of-sample stats
        '''

        df_lists = []
        for ticker, tts in tts_dict.items():

            df_test  = tts["test"]
            oos      = self.get_oos_stats(df_test[tts["y_col"]], df_test.predicted)
            df_add   = (self._get_model_params(
                tts["model"]).
                rename(columns = {"nobs": "train_nobs", "rsquared": "train_rsquared"}).
                assign(
                    ticker       = ticker,
                    train_end    = tts["train"].index.max(),
                    test_start   = df_test.index.min(),
                    oos_nobs     = int(oos.oos_nobs),
                    oos_r2       = oos.oos_r2,
                    oos_corr     = oos.oos_corr,
                    oos_hit_rate = oos.oos_hit_rate))

            df_lists.append(df_add)

        df_out = (pd.concat(
            df_lists)
            [["ticker", "variable", "params", "tvalue", "pvalue", "train_nobs",
              "train_rsquared", "train_end", "test_start", "oos_nobs", "oos_r2",
              "oos_corr", "oos_hit_rate"]].
            reset_index(drop = True))

        return df_out

    def fit_expanding(
            self,
            df        : pd.DataFrame,
            lag       : Union[int, Dict[str, int]] = 0,
            vol_norm  : bool = True,
            vol_window: int  = 100,
            vol_lag   : int  = 1) -> dict:

        '''
        Expanding window OLS per ticker with no min obs, starts at the first
        date OLS is estimable (3 obs). The predicted column at date t only uses
        params estimated through t-1 so it is out-of-sample

        Returns {ticker: DataFrame indexed by date}
        '''

        df_prep, y_col = self._prep_data(df, lag, vol_norm, vol_window, vol_lag)
        names          = ["const", self.signal_col]
        exp_dict       = {}

        for ticker, df_tmp in df_prep.groupby("ticker"):

            y = df_tmp[y_col].to_numpy(dtype = float)
            X = np.column_stack([
                np.ones(len(y)),
                df_tmp[self.signal_col].to_numpy(dtype = float)])

            df_stats = self._expanding_ols(y, X, names)
            df_stats.index = df_tmp.date.values

            exp_dict[ticker] = (df_tmp
                .set_index("date")
                [[self.signal_col, y_col]]
                .join(df_stats.rename_axis("date")))

        return exp_dict

# Testing on synthetic data only, no reading or writing

def _simulate_data(
        n_tickers: int   = 5,
        n_obs    : int   = 3_000,
        seed     : int   = 42) -> Tuple[pd.DataFrame, Dict[str, float]]:

    '''
    Simulated panel: persistent AR(1) signal (like an EWMA) and returns that
    load on it with a known beta, tickers start on different dates
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

def _check(name: str, condition: bool) -> bool:

    print("{:<55} {}".format(name, "PASS" if condition else "FAIL"))
    return condition

def main() -> None:

    df, true_betas = _simulate_data()
    fitter         = SignalFitter(signal_col = "signal", rtn_col = "px_rtn")
    results        = []

    print("Testing SignalFitter on simulated data\n")

    # 1. full sample recovers true betas (raw returns, no vol norm)
    start      = time.time()
    model_dict = fitter.fit_full_sample(df, vol_norm = False)
    df_params  = fitter.get_full_sample_params(model_dict)
    full_time  = time.time() - start

    df_beta = (df_params
        .loc[lambda x: x.variable == "signal"]
        .loc[lambda x: x.ticker.isin(list(true_betas.keys()))]
        .assign(
            true_beta = lambda x: x.ticker.map(true_betas),
            bse       = lambda x: x.params / x.tvalue,
            z_err     = lambda x: np.abs(x.params - x.true_beta) / x.bse))

    results.append(_check("Full sample betas within 4 SE of truth", (df_beta.z_err < 4).all()))

    # 2. vol_lag = 0 reproduces the notebook's vol_norm_rtn
    df_prep0, y_col = fitter._prep_data(df, vol_norm = True, vol_window = 100, vol_lag = 0)
    ticker          = "tick0_comdty"
    df_nb           = (df
        .loc[lambda x: x.ticker == ticker]
        .set_index("date")
        .assign(vol_norm_rtn = lambda x: x.px_rtn / x.px_rtn.ewm(span = 100, adjust = False).std())
        .dropna())

    df_mine = df_prep0.loc[lambda x: x.ticker == ticker].set_index("date")
    results.append(_check("vol_lag = 0 matches notebook vol_norm_rtn", np.allclose(
        df_nb.vol_norm_rtn.values, df_mine.loc[df_nb.index, "vol_norm_rtn"].values)))

    # 3. vol_lag = 1 uses only past vol
    df_prep1, _ = fitter._prep_data(df, vol_norm = True, vol_window = 100, vol_lag = 1)
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

    # 4. per ticker lag does not bleed across tickers
    lag_dict    = {"tick0_comdty": 2, "tick1_comdty": 5}
    df_lag, _   = fitter._prep_data(df, lag = lag_dict, vol_norm = False)
    lag_ok      = True
    for tmp_ticker in df.ticker.drop_duplicates():

        tmp_lag = lag_dict.get(tmp_ticker, 0)
        df_orig = df.loc[lambda x: x.ticker == tmp_ticker].sort_values("date").reset_index(drop = True)
        df_new  = df_lag.loc[lambda x: x.ticker == tmp_ticker].reset_index(drop = True)

        lag_ok  = lag_ok and (len(df_new) == len(df_orig) - tmp_lag)
        lag_ok  = lag_ok and np.allclose(df_new.signal.values, df_orig.signal.values[:len(df_orig) - tmp_lag])
        lag_ok  = lag_ok and (df_new.date.iloc[0] == df_orig.date.iloc[tmp_lag])

    results.append(_check("Per ticker lag dict shifts within ticker only", lag_ok))

    # 5. ratio split respects time order and ratio
    tts_ratio = fitter.fit_train_test_ratio(df, train_ratio = 0.7)
    ratio_ok  = all(
        (tts["train"].index.max() < tts["test"].index.min()) and
        (len(tts["train"]) == int(np.floor((len(tts["train"]) + len(tts["test"])) * 0.7)))
        for tts in tts_ratio.values())

    results.append(_check("Ratio split: time ordered, correct ratio", ratio_ok))

    # 6. date split respects cutoff and skips tickers with no train data
    cutoff = "2008-06-30"
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        tts_date = fitter.fit_train_test_date(df, cutoff_date = cutoff)

    date_ok = all(
        (tts["train"].index.max() <= pd.Timestamp(cutoff)) and
        (tts["test"].index.min() > pd.Timestamp(cutoff))
        for tts in tts_date.values())

    results.append(_check("Date split: train <= cutoff < test", date_ok))
    results.append(_check("Date split: late ticker skipped with warning",
        ("late_index" not in tts_date) and any("late_index" in str(w.message) for w in caught)))

    df_tts_params = fitter.get_train_test_params(tts_ratio)
    results.append(_check("Train / test params table built",
        (len(df_tts_params) == 2 * len(tts_ratio)) and df_tts_params.pvalue.notna().all()))

    # 7. expanding matches looped sm.OLS
    start    = time.time()
    exp_dict = fitter.fit_expanding(df)
    exp_time = time.time() - start

    df_exp       = exp_dict[ticker]
    check_points = [2, 3, 10, 250, 1_000, len(df_exp) - 1]
    exp_ok       = True

    start = time.time()
    for point in check_points:

        df_slice = df_exp.iloc[:point + 1]
        model    = (sm
            .OLS(
                endog = df_slice.vol_norm_rtn,
                exog  = sm.add_constant(df_slice.signal, has_constant = "add"))
            .fit())

        row    = df_exp.iloc[point]
        exp_ok = exp_ok and np.allclose(
            [row.const_param, row.signal_param, row.const_bse, row.signal_bse, row.signal_pvalue, row.rsquared],
            [model.params.const, model.params.signal, model.bse.const, model.bse.signal, model.pvalues.signal, model.rsquared])

    loop_time = (time.time() - start) / len(check_points)

    results.append(_check("Expanding matches sm.OLS at sample dates", exp_ok))
    results.append(_check("Expanding: first 2 rows NaN (no min obs)", df_exp.signal_param.iloc[:2].isna().all() and df_exp.signal_param.iloc[2:].notna().all()))

    # 8. expanding prediction uses only t-1 params
    pred_check = (
        df_exp.const_param.shift(1) +
        df_exp.signal_param.shift(1) * df_exp.signal)

    results.append(_check("Expanding predicted uses t-1 params", np.allclose(
        pred_check.dropna().values, df_exp.predicted.loc[pred_check.dropna().index].values)))

    # 9. last expanding row equals full sample fit
    full_vol   = fitter.fit_full_sample(df)[ticker]
    results.append(_check("Expanding last row equals full sample fit", np.allclose(
        [df_exp.signal_param.iloc[-1], df_exp.signal_tvalue.iloc[-1]],
        [full_vol.params.signal, full_vol.tvalues.signal])))

    n_total = sum(len(x) for x in exp_dict.values())

    print("\n{} / {} tests passed".format(sum(results), len(results)))
    print("\nTimings")
    print("Full sample fit + params ({} tickers): {:.3f}s".format(len(model_dict), full_time))
    print("Expanding fit ({} tickers, {:,} rows)  : {:.3f}s".format(len(exp_dict), n_total, exp_time))
    print("Looped sm.OLS estimate for same rows   : {:.1f}s".format(loop_time * n_total))

if __name__ == "__main__": main()
