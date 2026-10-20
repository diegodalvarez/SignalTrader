# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:49:00 2026

@author: Diego

Pure computation module for quantile (decile) conditioning of a signal.
Sorts the signal into n_bins buckets per ticker, measures the Sharpe of
returns in each bucket, and keeps the n_outer buckets on each side depending
on a Sharpe threshold and a group scaler. Nothing in here reads or writes to
disk

Expected input is a long DataFrame with columns
    date | ticker | <signal_col> | <rtn_col>

Positions are -1 / 0 / +1 per bucket, middle buckets are always 0. The
strategy return is signal_rtn = position * raw rtn_col, the final portfolio
volatility targeting is left to the notebook / Backtest
"""

import time
import warnings
import numpy as np
import pandas as pd

from   typing import Dict, Tuple, Union
from   SignalFitter import SignalFitter

class QuantileFitter(SignalFitter):

    def __init__(
            self,
            signal_col: str = "signal",
            rtn_col   : str = "px_rtn") -> None:

        super().__init__(signal_col = signal_col, rtn_col = rtn_col)

        self.ann_factor = np.sqrt(252)
        self.scalers    = {
            "sign": self._scaler_sign,
            "mean": self._scaler_mean}

    def _check_config(
            self,
            n_bins : int,
            n_outer: int,
            scaler : str) -> None:

        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")

        if (n_outer < 1) or (2 * n_outer > n_bins):
            raise ValueError("n_outer must be >= 1 and 2 * n_outer <= n_bins")

        if scaler not in self.scalers.keys():
            raise ValueError("scaler must be one of {}".format(list(self.scalers.keys())))

    def _sharpe(self, x: Union[pd.Series, np.ndarray]) -> float:

        x = np.asarray(x, dtype = float)
        x = x[np.isfinite(x)]

        if len(x) < 2:
            return np.nan

        std = np.std(x, ddof = 1)
        if std == 0:
            return np.nan

        return np.mean(x) / std * self.ann_factor

    def _get_bin_edges(self, signal: np.ndarray, n_bins: int) -> np.ndarray:

        '''
        Inner quantile edges (n_bins - 1 of them), the outer edges are
        implicitly -inf / +inf so out-of-sample values always land in a bin
        '''

        return np.quantile(signal, np.arange(1, n_bins) / n_bins)

    def _apply_bin_edges(self, signal: np.ndarray, edges: np.ndarray) -> np.ndarray:

        '''
        Right closed buckets like pd.qcut, values <= edges[0] go in bin 1
        '''

        return np.searchsorted(edges, signal, side = "left") + 1

    def _bins_full(self, signal: pd.Series, n_bins: int) -> np.ndarray:

        '''
        In-sample buckets, edges come from the ticker's full history
        '''

        values = signal.to_numpy(dtype = float)
        return self._apply_bin_edges(values, self._get_bin_edges(values, n_bins))

    def _bins_expanding(self, signal: pd.Series, n_bins: int) -> np.ndarray:

        '''
        Causal buckets, each signal is ranked against its own history through
        that date only (expanding percentile rank)
        '''

        pct = signal.expanding().rank(pct = True).to_numpy(dtype = float)
        return np.clip(np.ceil(pct * n_bins), 1, n_bins).astype(int)

    def _bin_sharpe(
            self,
            df    : pd.DataFrame,
            y_col : str,
            n_bins: int) -> pd.DataFrame:

        '''
        Sharpe per ticker per bin, every bin 1..n_bins shows up even if empty
        '''

        full_index = pd.MultiIndex.from_product(
            [df.ticker.drop_duplicates().sort_values(), np.arange(1, n_bins + 1)],
            names = ["ticker", "bin"])

        df_out = (df
            .groupby(["ticker", "bin"])
            [y_col]
            .agg(["mean", "std", "count"])
            .reindex(full_index)
            .assign(
                count  = lambda x: x["count"].fillna(0).astype(int),
                sharpe = lambda x: np.where(
                    (x["count"] >= 2) & (x["std"] > 0),
                    x["mean"] / x["std"] * self.ann_factor,
                    np.nan))
            .rename(columns = {"count": "nobs"})
            .reset_index())

        return df_out

    def _scaler_sign(
            self,
            sharpe   : np.ndarray,
            threshold: float) -> np.ndarray:

        '''
        Group is active only if every bin has the same sign and passes the
        absolute Sharpe threshold, each bin is traded with that sign. For
        n_outer = 2 this is the same as prod > 0
        sharpe is (n_rows, n_outer), returns positions of the same shape
        '''

        signs  = np.sign(sharpe)
        active = (
            np.all(np.isfinite(sharpe), axis = 1) &
            np.all(np.abs(sharpe) >= threshold, axis = 1) &
            np.all(signs == signs[:, [0]], axis = 1))

        return np.where(active[:, None], np.nan_to_num(signs), 0.0)

    def _scaler_mean(
            self,
            sharpe   : np.ndarray,
            threshold: float) -> np.ndarray:

        '''
        Group is active if the absolute mean Sharpe of its bins passes the
        threshold, every bin in the group is traded with the sign of the mean
        sharpe is (n_rows, n_outer), returns positions of the same shape
        '''

        valid  = np.all(np.isfinite(sharpe), axis = 1)
        mean   = np.where(valid, np.nanmean(np.where(valid[:, None], sharpe, 0.0), axis = 1), np.nan)
        active = valid & (np.abs(mean) >= threshold)
        pos    = np.where(active, np.sign(np.nan_to_num(mean)), 0.0)

        return np.repeat(pos[:, None], sharpe.shape[1], axis = 1)

    def _get_positions(
            self,
            sharpe          : np.ndarray,
            n_outer         : int,
            sharpe_threshold: float,
            scaler          : str) -> np.ndarray:

        '''
        sharpe is (n_rows, n_bins) and positions come back the same shape,
        rows can be tickers (in-sample / train) or dates (expanding)
        '''

        scaler_func = self.scalers[scaler]
        positions   = np.zeros_like(sharpe, dtype = float)

        positions[:, :n_outer] = scaler_func(sharpe[:, :n_outer], sharpe_threshold)
        positions[:, -n_outer:] = scaler_func(sharpe[:, -n_outer:], sharpe_threshold)

        return positions

    def _positions_table(
            self,
            df_sharpe       : pd.DataFrame,
            n_bins          : int,
            n_outer         : int,
            sharpe_threshold: float,
            scaler          : str) -> pd.DataFrame:

        df_wide   = df_sharpe.pivot(index = "ticker", columns = "bin", values = "sharpe")
        positions = self._get_positions(df_wide.to_numpy(dtype = float), n_outer, sharpe_threshold, scaler)

        df_out = (pd.DataFrame(
            data    = positions,
            index   = df_wide.index,
            columns = df_wide.columns).
            reset_index().
            melt(id_vars = "ticker", value_name = "position"))

        return df_out

    def _prep_quantile(
            self,
            df        : pd.DataFrame,
            lag       : Union[int, Dict[str, int]],
            vol_target: bool,
            vol_window: int,
            vol_lag   : int) -> Tuple[pd.DataFrame, str]:

        '''
        Lags the signal within ticker (so the bucket label is lagged) and
        optionally vol normalizes the returns used for the Sharpes. The raw
        rtn_col is kept for the strategy return
        '''

        return self._prep_data(df, lag, vol_target, vol_window, vol_lag)

    def _group_label(self, bins: pd.Series, n_bins: int, n_outer: int) -> np.ndarray:

        return np.where(
            bins <= n_outer, "lower",
            np.where(bins > n_bins - n_outer, "upper", "middle"))

    def fit_quantile_full_sample(
            self,
            df              : pd.DataFrame,
            n_bins          : int   = 10,
            n_outer         : int   = 2,
            sharpe_threshold: float = 0.0,
            scaler          : str   = "sign",
            lag             : Union[int, Dict[str, int]] = 1,
            vol_target      : bool  = True,
            vol_window      : int   = 100,
            vol_lag         : int   = 1) -> dict:

        '''
        In-sample quantile conditioning, bucket edges, Sharpes and positions
        all come from each ticker's full history

        Returns {"sharpe", "positions", "data", "y_col", "config"}
        '''

        self._check_config(n_bins, n_outer, scaler)
        df_prep, y_col = self._prep_quantile(df, lag, vol_target, vol_window, vol_lag)

        df_binned = (df_prep.assign(
            bin = lambda x: (x
                .groupby("ticker")
                [self.signal_col]
                .transform(lambda s: self._bins_full(s, n_bins))
                .astype(int))))

        df_sharpe    = self._bin_sharpe(df_binned, y_col, n_bins)
        df_positions = self._positions_table(df_sharpe, n_bins, n_outer, sharpe_threshold, scaler)

        df_data = (df_binned
            .merge(right = df_positions, how = "left", on = ["ticker", "bin"])
            .assign(signal_rtn = lambda x: x.position * x[self.rtn_col]))

        out_dict = {
            "sharpe"   : df_sharpe,
            "positions": df_positions,
            "data"     : df_data,
            "y_col"    : y_col,
            "config"   : {
                "n_bins"          : n_bins,
                "n_outer"         : n_outer,
                "sharpe_threshold": sharpe_threshold,
                "scaler"          : scaler}}

        return out_dict

    def get_quantile_params(self, fit_dict: dict) -> pd.DataFrame:

        '''
        One row per ticker per bin with Sharpe, nobs, position, group and
        whether that group was switched on
        '''

        n_bins  = fit_dict["config"]["n_bins"]
        n_outer = fit_dict["config"]["n_outer"]

        df_out = (fit_dict["sharpe"]
            .merge(right = fit_dict["positions"], how = "inner", on = ["ticker", "bin"])
            .assign(group = lambda x: self._group_label(x.bin, n_bins, n_outer))
            .assign(group_active = lambda x: (x
                .assign(tmp = lambda y: y.position != 0)
                .groupby(["ticker", "group"])
                .tmp
                .transform("any")
                .where(x.group != "middle", False)))
            [["ticker", "bin", "group", "mean", "std", "nobs", "sharpe", "position", "group_active"]])

        return df_out

    def _fit_quantile_split(
            self,
            df_train        : pd.DataFrame,
            df_test         : pd.DataFrame,
            y_col           : str,
            n_bins          : int,
            n_outer         : int,
            sharpe_threshold: float,
            scaler          : str) -> dict:

        '''
        Edges, Sharpes and positions come from train only, then are applied
        unchanged to test. Both frames are a single ticker with a date index
        '''

        edges = self._get_bin_edges(df_train[self.signal_col].to_numpy(dtype = float), n_bins)

        df_train_out = df_train.assign(bin = lambda x: self._apply_bin_edges(x[self.signal_col].to_numpy(dtype = float), edges))
        df_test_out  = df_test.assign(bin = lambda x: self._apply_bin_edges(x[self.signal_col].to_numpy(dtype = float), edges))

        df_sharpe    = self._bin_sharpe(df_train_out, y_col, n_bins)
        df_positions = self._positions_table(df_sharpe, n_bins, n_outer, sharpe_threshold, scaler)
        pos_map      = df_positions.set_index("bin").position

        df_train_out = df_train_out.assign(
            position   = lambda x: x.bin.map(pos_map).values,
            signal_rtn = lambda x: x.position * x[self.rtn_col])

        df_test_out = df_test_out.assign(
            position   = lambda x: x.bin.map(pos_map).values,
            signal_rtn = lambda x: x.position * x[self.rtn_col])

        out_dict = {
            "edges"    : edges,
            "sharpe"   : df_sharpe,
            "positions": df_positions,
            "train"    : df_train_out,
            "test"     : df_test_out,
            "y_col"    : y_col}

        return out_dict

    def fit_quantile_train_test_ratio(
            self,
            df              : pd.DataFrame,
            train_ratio     : float = 0.7,
            n_bins          : int   = 10,
            n_outer         : int   = 2,
            sharpe_threshold: float = 0.0,
            scaler          : str   = "sign",
            lag             : Union[int, Dict[str, int]] = 1,
            vol_target      : bool  = True,
            vol_window      : int   = 100,
            vol_lag         : int   = 1) -> dict:

        '''
        Train / test split by ratio respecting time order, each ticker is cut
        at its own ratio. Returns {ticker: {"edges", "sharpe", "positions",
        "train", "test", "y_col"}}
        '''

        if (train_ratio <= 0) or (train_ratio >= 1):
            raise ValueError("train_ratio must be between 0 and 1")

        self._check_config(n_bins, n_outer, scaler)
        df_prep, y_col = self._prep_quantile(df, lag, vol_target, vol_window, vol_lag)
        tts_dict       = {}
        skipped        = []

        for ticker, df_tmp in df_prep.groupby("ticker"):

            split = self._split_ratio(df_tmp.set_index("date"), train_ratio, n_bins)

            if split is None:
                skipped.append(ticker)
                continue

            tts_dict[ticker] = self._fit_quantile_split(
                split[0], split[1], y_col, n_bins, n_outer, sharpe_threshold, scaler)

        if len(skipped) > 0:
            warnings.warn("Skipped (too few train / test obs): {}".format(skipped), UserWarning, stacklevel = 2)

        return tts_dict

    def fit_quantile_train_test_date(
            self,
            df              : pd.DataFrame,
            cutoff_date     : Union[str, pd.Timestamp],
            n_bins          : int   = 10,
            n_outer         : int   = 2,
            sharpe_threshold: float = 0.0,
            scaler          : str   = "sign",
            lag             : Union[int, Dict[str, int]] = 1,
            vol_target      : bool  = True,
            vol_window      : int   = 100,
            vol_lag         : int   = 1) -> dict:

        '''
        Train / test split on a common date for every ticker
        train: date <= cutoff_date, test: date > cutoff_date
        '''

        self._check_config(n_bins, n_outer, scaler)
        cutoff         = pd.Timestamp(cutoff_date)
        df_prep, y_col = self._prep_quantile(df, lag, vol_target, vol_window, vol_lag)
        tts_dict       = {}
        skipped        = []

        for ticker, df_tmp in df_prep.groupby("ticker"):

            split = self._split_date(df_tmp.set_index("date"), cutoff, n_bins)

            if split is None:
                skipped.append(ticker)
                continue

            tts_dict[ticker] = self._fit_quantile_split(
                split[0], split[1], y_col, n_bins, n_outer, sharpe_threshold, scaler)

        if len(skipped) > 0:
            warnings.warn("Skipped (too few train / test obs): {}".format(skipped), UserWarning, stacklevel = 2)

        return tts_dict

    def get_quantile_train_test_stats(self, tts_dict: dict) -> pd.DataFrame:

        '''
        In-sample vs out-of-sample stats of signal_rtn per ticker. Hit rate is
        measured on days with a position only
        '''

        df_lists = []
        for ticker, tts in tts_dict.items():

            stats = {"ticker": ticker}
            for name in ["train", "test"]:

                df_tmp     = tts[name]
                active     = df_tmp.position != 0
                active_rtn = df_tmp.signal_rtn[active]

                stats[name + "_sharpe"]     = self._sharpe(df_tmp.signal_rtn)
                stats[name + "_hit_rate"]   = (active_rtn > 0).mean() if active.sum() > 0 else np.nan
                stats[name + "_pct_active"] = active.mean()
                stats[name + "_nobs"]       = len(df_tmp)

            stats["train_end"]  = tts["train"].index.max()
            stats["test_start"] = tts["test"].index.min()
            df_lists.append(stats)

        df_out = (pd.DataFrame(
            df_lists)
            [["ticker", "train_sharpe", "test_sharpe", "train_hit_rate", "test_hit_rate",
              "train_pct_active", "test_pct_active", "train_nobs", "test_nobs",
              "train_end", "test_start"]])

        return df_out

    def _expanding_bin_sharpe(
            self,
            y      : np.ndarray,
            bins   : np.ndarray,
            n_bins : int,
            min_obs: int) -> np.ndarray:

        '''
        Running Sharpe of every bin on every date in one pass, uses a one-hot
        matrix of returns and running sums of y, y^2 and counts. Row t only
        uses returns through t-1 (shifted one row) so it is out-of-sample
        Returns (n, n_bins)
        '''

        onehot = (bins[:, None] == np.arange(1, n_bins + 1)[None, :]).astype(float)
        y_hot  = onehot * y[:, None]

        cnt = np.cumsum(onehot, axis = 0)
        s1  = np.cumsum(y_hot, axis = 0)
        s2  = np.cumsum(y_hot * y[:, None], axis = 0)

        with np.errstate(divide = "ignore", invalid = "ignore"):
            mean   = s1 / cnt
            var    = np.maximum((s2 - cnt * mean ** 2) / (cnt - 1), 0)
            std    = np.sqrt(var)
            sharpe = np.where(
                (cnt >= max(min_obs, 2)) & (std > 0),
                mean / std * self.ann_factor,
                np.nan)

        sharpe_lag = np.full_like(sharpe, np.nan)
        sharpe_lag[1:] = sharpe[:-1]

        return sharpe_lag

    def fit_quantile_expanding(
            self,
            df              : pd.DataFrame,
            n_bins          : int   = 10,
            n_outer         : int   = 2,
            sharpe_threshold: float = 0.0,
            scaler          : str   = "sign",
            lag             : Union[int, Dict[str, int]] = 1,
            vol_target      : bool  = True,
            vol_window      : int   = 100,
            vol_lag         : int   = 1,
            min_obs         : int   = 2) -> dict:

        '''
        Expanding out-of-sample quantile conditioning per ticker
        1. bucket: signal ranked against its own history (lagged by lag)
        2. running Sharpe of every bucket through t-1
        3. positions from the scaler on those running Sharpes

        min_obs is the number of returns a bucket needs before it can take a
        position (2 is the minimum to have a std)

        Returns {ticker: DataFrame indexed by date}
        '''

        self._check_config(n_bins, n_outer, scaler)
        df_prep, y_col = self._prep_quantile(df, lag, vol_target, vol_window, vol_lag)
        sharpe_cols    = ["sharpe_{}".format(i) for i in range(1, n_bins + 1)]
        exp_dict       = {}

        for ticker, df_tmp in df_prep.groupby("ticker"):

            df_tmp = df_tmp.set_index("date").sort_index()
            bins   = self._bins_expanding(df_tmp[self.signal_col], n_bins)
            y      = df_tmp[y_col].to_numpy(dtype = float)

            sharpe    = self._expanding_bin_sharpe(y, bins, n_bins, min_obs)
            positions = self._get_positions(sharpe, n_outer, sharpe_threshold, scaler)
            position  = positions[np.arange(len(bins)), bins - 1]

            exp_dict[ticker] = (df_tmp
                [[self.signal_col, self.rtn_col, y_col]]
                .assign(
                    bin        = bins,
                    position   = position,
                    signal_rtn = lambda x: x.position * x[self.rtn_col])
                .join(pd.DataFrame(data = sharpe, index = df_tmp.index, columns = sharpe_cols)))

        return exp_dict

# Testing on synthetic data only, no reading or writing

def _simulate_quantile_data(
        n_tickers: int = 4,
        n_obs    : int = 3_000,
        seed     : int = 7) -> pd.DataFrame:

    '''
    Signal only predicts returns at its extremes (trend in the tails, noise in
    the middle), the last ticker is mean reverting in the tails
    '''

    rng      = np.random.default_rng(seed)
    dates    = pd.bdate_range(start = "2000-01-03", periods = n_obs)
    df_lists = []

    for i in range(n_tickers):

        start  = i * 150
        n      = n_obs - start
        signal = rng.normal(size = n)
        tail   = np.abs(signal) > np.quantile(np.abs(signal), 0.6)
        sign   = -1 if i == n_tickers - 1 else 1

        # signal is already shifted so it lines up with the return it predicts
        px_rtn = sign * 0.004 * np.sign(signal) * tail + rng.normal(scale = 0.01, size = n)

        df_lists.append(pd.DataFrame({
            "date"  : dates[start:],
            "ticker": "tick{}_comdty".format(i),
            "signal": signal,
            "px_rtn": px_rtn}))

    return pd.concat(df_lists).reset_index(drop = True)

def _check(name: str, condition: bool) -> bool:

    print("{:<60} {}".format(name, "PASS" if condition else "FAIL"))
    return condition

def main() -> None:

    df      = _simulate_quantile_data()
    fitter  = QuantileFitter(signal_col = "signal", rtn_col = "px_rtn")
    results = []
    trend   = "tick0_comdty"
    revert  = "tick3_comdty"

    print("Testing QuantileFitter on simulated data\n")

    # 1. full sample: trend tails long top / short bottom, middle flat
    start     = time.time()
    fit_dict  = fitter.fit_quantile_full_sample(df, lag = 0)
    df_params = fitter.get_quantile_params(fit_dict)
    full_time = time.time() - start

    pos_trend  = df_params.loc[lambda x: x.ticker == trend].set_index("bin").position
    pos_revert = df_params.loc[lambda x: x.ticker == revert].set_index("bin").position

    results.append(_check("Full sample: trend ticker -1 bottom, +1 top",
        (pos_trend.loc[[1, 2]] == -1).all() and (pos_trend.loc[[9, 10]] == 1).all()))
    results.append(_check("Full sample: mean reverting ticker flips signs",
        (pos_revert.loc[[1, 2]] == 1).all() and (pos_revert.loc[[9, 10]] == -1).all()))
    results.append(_check("Full sample: middle bins always 0",
        (df_params.loc[lambda x: x.group == "middle"].position == 0).all()))
    results.append(_check("Full sample: middle bin rows kept (not dropped)",
        len(fit_dict["data"]) == len(fitter._prep_data(df, 0, True, 100, 1)[0])))

    # 2. bins match pd.qcut
    df_tmp  = fit_dict["data"].loc[lambda x: x.ticker == trend]
    qcut    = pd.qcut(df_tmp.signal, q = 10, labels = False) + 1
    results.append(_check("Full sample bins match pd.qcut", (qcut.values == df_tmp.bin.values).all()))

    # 3. lag = 1 lags the bucket within ticker
    fit_lag = fitter.fit_quantile_full_sample(df, lag = 1, vol_target = False)
    fit_raw = fitter.fit_quantile_full_sample(df, lag = 0, vol_target = False)
    n_lag   = fit_lag["data"].groupby("ticker").size()
    n_none  = fit_raw["data"].groupby("ticker").size()
    results.append(_check("lag = 1 drops exactly one row per ticker", ((n_none - n_lag) == 1).all()))

    # 4. threshold zeroes everything when set very high
    fit_high = fitter.fit_quantile_full_sample(df, lag = 0, sharpe_threshold = 50)
    results.append(_check("Huge |Sharpe| threshold zeroes all positions", (fit_high["positions"].position == 0).all()))

    # 5. scaler unit tests on crafted Sharpes (n_outer = 3)
    crafted = np.array([
        [-1.0, -0.5,  0.8],    # prod > 0 but mixed signs
        [ 0.6,  0.4,  0.9],    # all positive
        [-0.2,  2.0,  0.1],    # mixed, mean positive
        [ 0.6,  0.1,  0.9]])   # all positive, one below threshold

    pos_sign = fitter._scaler_sign(crafted, threshold = 0.3)
    pos_mean = fitter._scaler_mean(crafted, threshold = 0.3)

    results.append(_check("Sign scaler: mixed signs off even when prod > 0", (pos_sign[0] == 0).all()))
    results.append(_check("Sign scaler: same sign above threshold on", (pos_sign[1] == 1).all()))
    results.append(_check("Sign scaler: one bin under |threshold| turns group off", (pos_sign[3] == 0).all()))
    results.append(_check("Mean scaler: trades sign of mean for whole group",
        (pos_mean[0] == 0).all() and (pos_mean[1] == 1).all() and (pos_mean[2] == 1).all()))

    # 6. train / test: nothing from test leaks into the fit
    cutoff   = "2006-12-29"
    tts_date = fitter.fit_quantile_train_test_date(df, cutoff_date = cutoff)
    df_shock = df.assign(px_rtn = lambda x: np.where(x.date > pd.Timestamp(cutoff), -x.px_rtn * 50, x.px_rtn))
    tts_shck = fitter.fit_quantile_train_test_date(df_shock, cutoff_date = cutoff)

    leak_ok = all(
        np.allclose(tts_date[t]["edges"], tts_shck[t]["edges"]) and
        tts_date[t]["positions"].position.equals(tts_shck[t]["positions"].position)
        for t in tts_date.keys())

    results.append(_check("Date split: edges / positions ignore test data", leak_ok))

    edges_ok = all(
        np.allclose(tts["edges"], np.quantile(tts["train"].signal, np.arange(1, 10) / 10))
        for tts in tts_date.values())
    results.append(_check("Date split: edges are train quantiles", edges_ok))

    tts_ratio = fitter.fit_quantile_train_test_ratio(df, train_ratio = 0.7, scaler = "mean")
    ratio_ok  = all(tts["train"].index.max() < tts["test"].index.min() for tts in tts_ratio.values())
    results.append(_check("Ratio split: time ordered", ratio_ok))

    df_stats = fitter.get_quantile_train_test_stats(tts_ratio)
    results.append(_check("Train / test stats: trend ticker positive OOS Sharpe",
        df_stats.set_index("ticker").loc[trend, "test_sharpe"] > 0))

    # 7. expanding: running Sharpe and bins match brute force
    start    = time.time()
    exp_dict = fitter.fit_quantile_expanding(df, lag = 0)
    exp_time = time.time() - start

    df_exp   = exp_dict[trend]
    brute_ok = True
    for point in [5, 50, 500, len(df_exp) - 1]:

        df_hist  = df_exp.iloc[:point]
        pct      = (df_exp.signal.iloc[:point + 1].rank(pct = True)).iloc[-1]
        brute_ok = brute_ok and (df_exp.bin.iloc[point] == int(np.clip(np.ceil(pct * 10), 1, 10)))

        for b in range(1, 11):

            sharpe_b = fitter._sharpe(df_hist.loc[df_hist.bin == b, "vol_norm_rtn"])
            mine     = df_exp["sharpe_{}".format(b)].iloc[point]
            brute_ok = brute_ok and ((np.isnan(sharpe_b) and np.isnan(mine)) or np.isclose(sharpe_b, mine))

    results.append(_check("Expanding: bins and running Sharpes match brute force", brute_ok))

    # 8. expanding: changing the future never changes past positions
    k        = 1_500
    df_fut   = df.copy()
    late     = df_fut.groupby("ticker").cumcount() > k
    df_fut.loc[late, ["signal", "px_rtn"]] = df_fut.loc[late, ["signal", "px_rtn"]] * -3
    exp_fut  = fitter.fit_quantile_expanding(df_fut, lag = 0)

    first_changed = df_fut.loc[late].groupby("ticker").date.min()
    future_ok     = all(
        np.array_equal(
            exp_dict[t].loc[lambda x: x.index < first_changed[t]].position.values,
            exp_fut[t].loc[lambda x: x.index < first_changed[t]].position.values)
        for t in exp_dict.keys())

    results.append(_check("Expanding: past positions unaffected by future data", future_ok))
    results.append(_check("Expanding: trend ticker positive Sharpe", fitter._sharpe(df_exp.signal_rtn) > 0))

    n_total = sum(len(x) for x in exp_dict.values())

    print("\n{} / {} tests passed".format(sum(results), len(results)))
    print("\nTimings")
    print("Full sample fit + params ({} tickers)   : {:.3f}s".format(len(fit_dict["positions"].ticker.unique()), full_time))
    print("Expanding fit ({} tickers, {:,} rows)   : {:.3f}s".format(len(exp_dict), n_total, exp_time))

if __name__ == "__main__": main()
