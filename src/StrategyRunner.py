# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:00:00 2026

@author: Diego

Runs every model of a fitter on one signal, holds the results in one object
and applies swappable volatility targets per signal. Nothing in here reads or
writes to disk, DataFrames go in and DataFrames come out

Two parts
    1. Vol targets: objects (or plain functions) with one method
           get_leverage(df_signal_rtn, df_asset_rtn) -> leverage
       Inputs are wide (date x name) for one model, leverage comes back the
       same shape. lag = 1 uses vol through t-1 (realistic hedging), lag = 0
       uses vol through t (instantaneous hedging)

    2. StrategyRunner: fits full_sample, train_test and expanding once with
       run(), then apply(vol_target) as many times as needed without
       refitting. apply returns and stores a long frame

           <date_col> | <group_col> | model | sample | vol_target | signal |
           <rtn_col> | weight | leverage | signal_rtn

       <rtn_col> is the asset return, weight is the position, signal_rtn =
       weight * rtn before leverage, so
       the vol targeted return is leverage * signal_rtn. Leverage is kept
       separate so it can be capped or tweaked before averaging

Volatility targeting is per signal only, the equal weight average is a
convenience, portfolio construction lives elsewhere
"""

import time
import inspect
import warnings
import numpy as np
import pandas as pd

from   typing import Callable, Dict, List, Tuple, Union

class VolTarget:

    '''
    Base vol target, subclasses only define _get_vol (daily vol, unlagged)

    target      : annualized vol target
    lag         : rows to shift the vol estimate, 1 = realistic, 0 = instantaneous
    vol_on      : "signal_rtn" (vol of the strategy return) or "rtn" (vol of the asset)
    max_leverage: optional cap on leverage, None = no cap
    '''

    def __init__(
            self,
            target      : float = 0.1,
            lag         : int   = 1,
            vol_on      : str   = "signal_rtn",
            max_leverage: Union[float, None] = None) -> None:

        if vol_on not in ["signal_rtn", "rtn"]:
            raise ValueError("vol_on must be one of ['signal_rtn', 'rtn']")

        if lag < 0:
            raise ValueError("lag must be >= 0")

        self.target       = target
        self.lag          = lag
        self.vol_on       = vol_on
        self.max_leverage = max_leverage
        self.ann_factor   = np.sqrt(252)

    def _get_vol(self, df_rtn: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _name_body(self) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:

        name = "{}_{:g}pct_lag{}".format(self._name_body(), self.target * 100, self.lag)

        if self.vol_on == "rtn":
            name += "_assetvol"

        if self.max_leverage is not None:
            name += "_cap{:g}".format(self.max_leverage)

        return name

    def get_leverage(
            self,
            df_signal_rtn: pd.DataFrame,
            df_asset_rtn : pd.DataFrame) -> pd.DataFrame:

        df_rtn   = df_signal_rtn if self.vol_on == "signal_rtn" else df_asset_rtn
        df_vol   = self._get_vol(df_rtn).shift(self.lag)
        leverage = (self.target / (df_vol * self.ann_factor)).replace([np.inf, -np.inf], np.nan)

        if self.max_leverage is not None:
            leverage = leverage.clip(upper = self.max_leverage)

        return leverage

    def __repr__(self) -> str:
        return "{}(name = {})".format(type(self).__name__, self.name)

class EWMVolTarget(VolTarget):

    def __init__(
            self,
            span        : int   = 100,
            target      : float = 0.1,
            lag         : int   = 1,
            vol_on      : str   = "signal_rtn",
            max_leverage: Union[float, None] = None) -> None:

        super().__init__(target = target, lag = lag, vol_on = vol_on, max_leverage = max_leverage)
        self.span = span

    def _get_vol(self, df_rtn: pd.DataFrame) -> pd.DataFrame:
        return df_rtn.ewm(span = self.span, adjust = False).std()

    def _name_body(self) -> str:
        return "ewm_{}".format(self.span)

class RollingVolTarget(VolTarget):

    def __init__(
            self,
            window      : int   = 63,
            target      : float = 0.1,
            lag         : int   = 1,
            vol_on      : str   = "signal_rtn",
            max_leverage: Union[float, None] = None,
            min_periods : Union[int, None]   = None) -> None:

        super().__init__(target = target, lag = lag, vol_on = vol_on, max_leverage = max_leverage)
        self.window      = window
        self.min_periods = min_periods

    def _get_vol(self, df_rtn: pd.DataFrame) -> pd.DataFrame:
        return df_rtn.rolling(window = self.window, min_periods = self.min_periods).std()

    def _name_body(self) -> str:
        return "rolling_{}".format(self.window)

class NoVolTarget(VolTarget):

    '''
    Leverage of 1 wherever there is a return, checks the plumbing
    '''

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "no_vol_target"

    def get_leverage(
            self,
            df_signal_rtn: pd.DataFrame,
            df_asset_rtn : pd.DataFrame) -> pd.DataFrame:

        return df_signal_rtn.notna().astype(float).where(df_signal_rtn.notna())

class StrategyRunner:

    '''
    Fits every model of a fitter on one signal and applies vol targets

    fitter     : a QuantileFitter (its group_col, date_col, rtn_col are used)
    signal_col : the signal to fit
    models     : any of ("full_sample", "train_test", "expanding")
    split      : "date" (default) or "ratio" for the train_test model
    cutoff_date: train <= cutoff_date < test, None infers it (split = "date")
    train_ratio: used by split = "ratio" and to infer the cutoff date
    min_train_obs: warn if a name has fewer train rows than this
    fit_kwargs : passed to every fit (n_bins, n_outer, scaler, lag, ...),
                 each fit only gets the arguments it accepts
    '''

    def __init__(
            self,
            fitter       : object,
            signal_col   : str,
            models       : Tuple[str, ...] = ("full_sample", "train_test", "expanding"),
            split        : str   = "date",
            cutoff_date  : Union[str, pd.Timestamp, None] = None,
            train_ratio  : float = 0.7,
            min_train_obs: int   = 252,
            fit_kwargs   : Union[dict, None] = None) -> None:

        all_models = ["full_sample", "train_test", "expanding"]

        if len(models) == 0 or any(model not in all_models for model in models):
            raise ValueError("models must be a non empty subset of {}".format(all_models))

        if split not in ["date", "ratio"]:
            raise ValueError("split must be one of ['date', 'ratio']")

        if (train_ratio <= 0) or (train_ratio >= 1):
            raise ValueError("train_ratio must be between 0 and 1")

        self.fitter        = fitter
        self.signal_col    = signal_col
        self.models        = tuple(models)
        self.split         = split
        self.cutoff_date   = None if cutoff_date is None else pd.Timestamp(cutoff_date)
        self.train_ratio   = train_ratio
        self.min_train_obs = min_train_obs
        self.fit_kwargs    = {} if fit_kwargs is None else dict(fit_kwargs)

        self.group_col  = fitter.group_col
        self.date_col   = fitter.date_col
        self.rtn_col    = fitter.rtn_col
        self.ann_factor = np.sqrt(252)

        self.fits        = {}
        self.signal_data = None
        self.results     = {}
        self.config      = {}

    def __repr__(self) -> str:

        return "StrategyRunner(signal_col = {}, models = {}, split = {}, cutoff_date = {}, ran = {}, vol_targets = {})".format(
            self.signal_col, list(self.models), self.split, self.config.get("cutoff_date"),
            self.signal_data is not None, list(self.results.keys()))

    def _infer_cutoff(self, df: pd.DataFrame) -> pd.Timestamp:

        '''
        Common window runs from the latest first date across names to the last
        date, the cutoff is train_ratio of the way through it, on a real date
        '''

        dates      = pd.to_datetime(df[self.date_col])
        first_date = dates.groupby(df[self.group_col]).min()
        start      = first_date.max()
        window     = np.sort(dates.loc[dates >= start].unique())
        cutoff     = pd.Timestamp(window[int(np.floor((len(window) - 1) * self.train_ratio))])

        return cutoff

    def _check_train_obs(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> None:

        n_train = (pd.to_datetime(df[self.date_col]) <= cutoff).groupby(df[self.group_col]).sum()
        short   = n_train.loc[lambda x: x < self.min_train_obs]

        if len(short) > 0:
            warnings.warn(
                "Fewer than {} train rows before {}: {}".format(
                    self.min_train_obs, cutoff.date(), short.to_dict()),
                UserWarning, stacklevel = 3)

    def _call_fit(self, method: Callable, df: pd.DataFrame, **kwargs) -> dict:

        '''
        Passes fit_kwargs plus kwargs, keeping only what the method accepts
        '''

        accepted = inspect.signature(method).parameters
        all_args = {**self.fit_kwargs, **kwargs}

        return method(df, self.signal_col, **{k: v for k, v in all_args.items() if k in accepted})

    def _fit_models(self, df: pd.DataFrame) -> dict:

        fits = {}

        if "full_sample" in self.models:
            fits["full_sample"] = self._call_fit(self.fitter.fit_quantile_full_sample, df)

        if "train_test" in self.models:

            if self.split == "date":
                fits["train_test"] = self._call_fit(
                    self.fitter.fit_quantile_train_test_date, df, cutoff_date = self.config["cutoff_date"])

            else:
                fits["train_test"] = self._call_fit(
                    self.fitter.fit_quantile_train_test_ratio, df, train_ratio = self.train_ratio)

        if "expanding" in self.models:
            fits["expanding"] = self._call_fit(self.fitter.fit_quantile_expanding, df)

        return fits

    def run(self, df: pd.DataFrame) -> "StrategyRunner":

        '''
        Fits every model once, stores the raw fits, the stacked long frame
        (signal_data) and the config. Returns self
        '''

        cutoff_inferred = False
        cutoff          = None

        if ("train_test" in self.models) and (self.split == "date"):

            cutoff = self.cutoff_date
            if cutoff is None:
                cutoff          = self._infer_cutoff(df)
                cutoff_inferred = True

            self._check_train_obs(df, cutoff)

        self.config = {
            "signal_col"     : self.signal_col,
            "group_col"      : self.group_col,
            "date_col"       : self.date_col,
            "rtn_col"        : self.rtn_col,
            "models"         : list(self.models),
            "split"          : self.split if "train_test" in self.models else None,
            "cutoff_date"    : cutoff,
            "cutoff_inferred": cutoff_inferred,
            "train_ratio"    : self.train_ratio,
            "fit_kwargs"     : dict(self.fit_kwargs)}

        self.fits = self._fit_models(df)

        self.signal_data = (pd.concat([
            self.fitter.get_quantile_data(fit_output).assign(model = model)
            for model, fit_output in self.fits.items()])
            .reset_index(drop = True))

        first_cols       = [self.date_col, self.group_col, "model", "sample"]
        self.signal_data = (self.signal_data
            [first_cols + [col for col in self.signal_data.columns if col not in first_cols]]
            .sort_values(["model", self.group_col, self.date_col])
            .reset_index(drop = True))

        self.results = {}
        return self

    def _vol_name(self, vol_target: Union[VolTarget, Callable]) -> str:

        if isinstance(vol_target, VolTarget):
            return vol_target.name

        if callable(vol_target):
            return vol_target.__name__

        raise TypeError("vol_target must be a VolTarget or a function (df_signal_rtn, df_asset_rtn) -> leverage")

    def _get_leverage(
            self,
            vol_target   : Union[VolTarget, Callable],
            df_signal_rtn: pd.DataFrame,
            df_asset_rtn : pd.DataFrame) -> pd.DataFrame:

        if isinstance(vol_target, VolTarget):
            leverage = vol_target.get_leverage(df_signal_rtn, df_asset_rtn)

        else:
            leverage = vol_target(df_signal_rtn, df_asset_rtn)

        if (isinstance(leverage, pd.DataFrame) == False) or (leverage.shape != df_signal_rtn.shape):
            raise ValueError("vol_target must return a DataFrame shaped like df_signal_rtn")

        return leverage.replace([np.inf, -np.inf], np.nan)

    def apply(self, vol_target: Union[VolTarget, Callable]) -> pd.DataFrame:

        '''
        Leverage per signal for every model, no refitting. Stores and returns

            <date_col> | <group_col> | model | sample | vol_target | signal |
            <rtn_col> | weight | leverage | signal_rtn

        <rtn_col> is the asset return, signal_rtn = weight * <rtn_col>,
        vol targeted return = leverage * signal_rtn
        '''

        if self.signal_data is None:
            raise ValueError("call run(df) before apply")

        vol_name = self._vol_name(vol_target)
        if vol_name in self.results:
            warnings.warn("Overwriting results for vol target {}".format(vol_name), UserWarning, stacklevel = 2)

        df_lists = []
        for model, df_model in self.signal_data.groupby("model", sort = False):

            df_signal_rtn = df_model.pivot(index = self.date_col, columns = self.group_col, values = "signal_rtn")
            df_asset_rtn  = df_model.pivot(index = self.date_col, columns = self.group_col, values = self.rtn_col)

            df_leverage = (self
                ._get_leverage(vol_target, df_signal_rtn, df_asset_rtn)
                .rename_axis(index = self.date_col, columns = None)
                .reset_index()
                .melt(id_vars = self.date_col, var_name = self.group_col, value_name = "leverage"))

            df_add = (df_model
                [[self.date_col, self.group_col, "model", "sample", self.signal_col, self.rtn_col, "position", "signal_rtn"]]
                .merge(right = df_leverage, how = "left", on = [self.date_col, self.group_col]))

            df_lists.append(df_add)

        df_out = (pd.concat(
            df_lists)
            .rename(columns = {self.signal_col: "signal", "position": "weight"})
            .assign(vol_target = vol_name)
            [[self.date_col, self.group_col, "model", "sample", "vol_target", "signal", self.rtn_col, "weight", "leverage", "signal_rtn"]]
            .reset_index(drop = True))

        self.results[vol_name] = df_out
        return df_out

    def get_equal_weight(self, vol_targets: Union[List[str], None] = None) -> pd.DataFrame:

        '''
        Equal weight average of leverage * signal_rtn across names per date

            <date_col> | model | sample | vol_target | port_rtn

        Names with no leverage on a date (e.g. vol not estimable yet) are left
        out of that date's average
        '''

        names = list(self.results.keys()) if vol_targets is None else vol_targets
        if len(names) == 0:
            raise ValueError("no vol targets applied yet, call apply(vol_target)")

        df_out = (pd.concat([
            self.results[name] for name in names])
            .assign(vol_rtn = lambda x: x.leverage * x.signal_rtn)
            .groupby([self.date_col, "model", "sample", "vol_target"])
            .vol_rtn
            .mean()
            .rename("port_rtn")
            .reset_index())

        return df_out

    def _perf_stats(self, x: pd.Series) -> dict:

        x = x.dropna()
        if len(x) < 2:
            return {"nobs": len(x), "ann_rtn": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}

        wealth = np.cumprod(1 + x)
        std    = x.std()

        stats = {
            "nobs"        : len(x),
            "ann_rtn"     : x.mean() * 252,
            "ann_vol"     : std * self.ann_factor,
            "sharpe"      : x.mean() / std * self.ann_factor if std > 0 else np.nan,
            "max_drawdown": (wealth / wealth.cummax() - 1).min()}

        return stats

    def get_stats(
            self,
            level      : str = "equal_weight",
            vol_targets: Union[List[str], None] = None) -> pd.DataFrame:

        '''
        Annualized return, vol, Sharpe, max drawdown (compounded), nobs,
        start, end for every applied vol target
            level = "equal_weight": per model x sample x vol_target
            level = "signal"      : per name x model x sample x vol_target
        '''

        if level == "equal_weight":

            df_rtn = self.get_equal_weight(vol_targets).rename(columns = {"port_rtn": "vol_rtn"})
            keys   = ["model", "sample", "vol_target"]

        elif level == "signal":

            names  = list(self.results.keys()) if vol_targets is None else vol_targets
            df_rtn = (pd.concat([
                self.results[name] for name in names])
                .assign(vol_rtn = lambda x: x.leverage * x.signal_rtn))
            keys   = [self.group_col, "model", "sample", "vol_target"]

        else:
            raise ValueError("level must be one of ['equal_weight', 'signal']")

        rows = []
        for key, df_tmp in df_rtn.sort_values(self.date_col).groupby(keys):

            row = dict(zip(keys, key))
            row.update(self._perf_stats(df_tmp.vol_rtn))
            row.update({
                "start": df_tmp[self.date_col].min(),
                "end"  : df_tmp[self.date_col].max()})

            rows.append(row)

        return pd.DataFrame(rows)

# Testing on synthetic data only, no reading or writing

def _check(name: str, condition: bool) -> bool:

    print("{:<62} {}".format(name, "PASS" if condition else "FAIL"))
    return bool(condition)

def main() -> None:

    from QuantileFitter import QuantileFitter, _simulate_quantile_data

    df      = _simulate_quantile_data()
    fitter  = QuantileFitter(rtn_col = "px_rtn")
    results = []
    name    = "tick0_comdty"

    print("Testing StrategyRunner on simulated data\n")

    # 1. run fits every model once, default date split with an inferred cutoff
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner = StrategyRunner(fitter, "signal").run(df)
    run_time = time.time() - start

    cutoff      = runner.config["cutoff_date"]
    first_dates = df.groupby("ticker").date.min()

    results.append(_check("run: one fit per model, signal_data has every model",
        (set(runner.fits.keys()) == {"full_sample", "train_test", "expanding"}) and
        (set(runner.signal_data.model) == {"full_sample", "train_test", "expanding"})))
    results.append(_check("run: signal_data matches get_quantile_data row for row", all(
        len(runner.signal_data.loc[lambda x: x.model == model]) == len(fitter.get_quantile_data(fit_output))
        for model, fit_output in runner.fits.items())))
    results.append(_check("run: default split is date, cutoff inferred and recorded",
        (runner.config["split"] == "date") and runner.config["cutoff_inferred"] and isinstance(cutoff, pd.Timestamp)))

    # 2. inferred cutoff sits in the common window and is used by train_test
    window = np.sort(df.loc[df.date >= first_dates.max()].date.unique())
    df_tt  = runner.signal_data.loc[lambda x: x.model == "train_test"]

    results.append(_check("Cutoff: train_ratio of the common window, on a real date",
        (cutoff == pd.Timestamp(window[int(np.floor((len(window) - 1) * 0.7))])) and (cutoff in set(df.date))))
    results.append(_check("Cutoff: train <= cutoff < test for every name",
        (df_tt.loc[lambda x: x["sample"] == "train"].date.max() <= cutoff) and
        (df_tt.loc[lambda x: x["sample"] == "test"].date.min() > cutoff) and
        (df_tt.groupby("ticker")["sample"].nunique() == 2).all()))

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        StrategyRunner(fitter, "signal", models = ("train_test",), min_train_obs = 10_000).run(df)

    results.append(_check("Cutoff: warns when a name has too few train rows",
        any("Fewer than 10000 train rows" in str(w.message) for w in caught)))

    # 3. explicit cutoff and ratio split
    runner_dt = StrategyRunner(fitter, "signal", models = ("train_test",), cutoff_date = "2008-06-30", min_train_obs = 0).run(df)
    runner_rt = StrategyRunner(fitter, "signal", models = ("train_test",), split = "ratio").run(df)
    tts_ratio = fitter.fit_quantile_train_test_ratio(df, "signal", train_ratio = 0.7)

    results.append(_check("Explicit cutoff is used, not inferred",
        (runner_dt.config["cutoff_date"] == pd.Timestamp("2008-06-30")) and (runner_dt.config["cutoff_inferred"] == False) and
        (runner_dt.signal_data.loc[lambda x: x["sample"] == "train"].date.max() <= pd.Timestamp("2008-06-30"))))
    results.append(_check("Ratio split matches fit_quantile_train_test_ratio",
        (runner_rt.config["cutoff_date"] is None) and
        np.allclose(runner_rt.signal_data.signal_rtn.values, fitter.get_quantile_data(tts_ratio).signal_rtn.values)))

    # 4. fit_kwargs only go to the fits that accept them
    runner_kw = StrategyRunner(fitter, "signal", min_train_obs = 0, fit_kwargs = {"n_bins": 5, "n_outer": 1, "min_obs": 5}).run(df)
    results.append(_check("fit_kwargs: n_bins reaches every model, min_obs only expanding",
        (runner_kw.signal_data.bin.max() == 5) and ("sharpe_5" in runner_kw.signal_data.columns) and
        ("sharpe_6" not in runner_kw.signal_data.columns)))

    # 5. output format
    start  = time.time()
    df_ewm = runner.apply(EWMVolTarget(span = 100))
    apply_time = time.time() - start

    results.append(_check("apply: long format with the agreed columns", list(df_ewm.columns) ==
        ["date", "ticker", "model", "sample", "vol_target", "signal", "px_rtn", "weight", "leverage", "signal_rtn"]))
    results.append(_check("apply: rows match, weight = position, signal_rtn = weight * rtn",
        (len(df_ewm) == len(runner.signal_data)) and
        np.array_equal(df_ewm.weight.values, runner.signal_data.position.values) and
        np.allclose(df_ewm.px_rtn.values, runner.signal_data.px_rtn.values) and
        np.allclose(df_ewm.signal_rtn.values, (df_ewm.weight * df_ewm.px_rtn).values) and
        np.allclose(df_ewm.signal_rtn.values, runner.signal_data.signal_rtn.values)))

    # 6. NoVolTarget leaves returns unchanged
    df_none = runner.apply(NoVolTarget())
    results.append(_check("NoVolTarget: leverage 1, vol return = signal_rtn",
        (df_none.leverage == 1).all() and np.allclose((df_none.leverage * df_none.signal_rtn).values, df_none.signal_rtn.values)))

    # 7. EWM matches the notebook formula, with and without the lag
    wide   = runner.signal_data.loc[lambda x: x.model == "full_sample"].pivot(index = "date", columns = "ticker", values = "signal_rtn")
    manual = (0.1 / (wide.ewm(span = 100, adjust = False).std() * np.sqrt(252))).replace([np.inf, -np.inf], np.nan)
    mine1  = df_ewm.loc[lambda x: (x.model == "full_sample") & (x.ticker == name)].set_index("date").leverage

    df_ewm0 = runner.apply(EWMVolTarget(span = 100, lag = 0))
    mine0   = df_ewm0.loc[lambda x: (x.model == "full_sample") & (x.ticker == name)].set_index("date").leverage

    results.append(_check("EWM lag = 1: 0.1 / (ewm std shifted one day * sqrt 252)",
        np.allclose(mine1.values, manual[name].shift(1).loc[mine1.index].values, equal_nan = True)))
    results.append(_check("EWM lag = 0: same without the shift (instantaneous)",
        np.allclose(mine0.values, manual[name].loc[mine0.index].values, equal_nan = True)))

    # 8. lag = 1 never uses day t, lag = 0 does
    vol_tgt  = EWMVolTarget(span = 20)
    df_bump  = wide.copy()
    bump_row = 500
    df_bump.iloc[bump_row] = df_bump.iloc[bump_row] * 50

    lev_base = vol_tgt.get_leverage(wide, wide)
    lev_bump = vol_tgt.get_leverage(df_bump, df_bump)
    lev0     = EWMVolTarget(span = 20, lag = 0)

    results.append(_check("lag = 1: leverage on day t ignores day t's return",
        np.allclose(lev_base.iloc[:bump_row + 1].values, lev_bump.iloc[:bump_row + 1].values, equal_nan = True)))
    results.append(_check("lag = 0: leverage on day t uses day t's return", not np.allclose(
        lev0.get_leverage(wide, wide).iloc[bump_row].values, lev0.get_leverage(df_bump, df_bump).iloc[bump_row].values, equal_nan = True)))

    # 9. cap, asset vol, rolling, inf handling, plain functions
    df_cap = runner.apply(EWMVolTarget(span = 100, max_leverage = 2))
    results.append(_check("max_leverage caps leverage, default has no cap",
        (df_cap.leverage.max() <= 2) and (df_ewm.leverage.max() > 2)))

    df_asset   = runner.apply(EWMVolTarget(span = 100, vol_on = "rtn"))
    wide_asset = runner.signal_data.loc[lambda x: x.model == "full_sample"].pivot(index = "date", columns = "ticker", values = "px_rtn")
    manual_a   = 0.1 / (wide_asset.ewm(span = 100, adjust = False).std().shift(1) * np.sqrt(252))
    mine_a     = df_asset.loc[lambda x: (x.model == "full_sample") & (x.ticker == name)].set_index("date").leverage
    results.append(_check("vol_on = rtn uses the asset's own vol",
        np.allclose(mine_a.values, manual_a[name].loc[mine_a.index].values, equal_nan = True)))

    df_roll = runner.apply(RollingVolTarget(window = 63))
    manual_r = 0.1 / (wide.rolling(63).std().shift(1) * np.sqrt(252))
    mine_r   = df_roll.loc[lambda x: (x.model == "full_sample") & (x.ticker == name)].set_index("date").leverage
    results.append(_check("RollingVolTarget matches rolling std shifted",
        np.allclose(mine_r.values, manual_r[name].loc[mine_r.index].values, equal_nan = True)))

    df_flat = pd.DataFrame({"a": [0.0] * 5 + [0.01, -0.01, 0.02], "b": np.linspace(-0.01, 0.01, 8)})
    lev_flat = EWMVolTarget(span = 3, lag = 0).get_leverage(df_flat, df_flat)
    results.append(_check("Zero vol gives NaN leverage, never inf", np.isfinite(lev_flat.fillna(0).values).all() and lev_flat.a.iloc[:5].isna().all()))

    def flat_two(df_signal_rtn: pd.DataFrame, df_asset_rtn: pd.DataFrame) -> pd.DataFrame:
        return df_signal_rtn * 0 + 2

    df_func = runner.apply(flat_two)
    results.append(_check("Plain function works as a vol target, named after it",
        ("flat_two" in runner.results) and (df_func.leverage.dropna() == 2).all()))

    # 10. results store every vol target, overwriting warns
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        runner.apply(NoVolTarget())

    results.append(_check("results keep every vol target, re-applying warns",
        (len(runner.results) == 7) and any("Overwriting" in str(w.message) for w in caught)))

    # 11. equal weight and stats
    df_eq   = runner.get_equal_weight([EWMVolTarget(span = 100).name])
    df_day  = df_ewm.loc[lambda x: (x.model == "expanding") & (x.date == x.date.max())]
    eq_day  = df_eq.loc[lambda x: (x.model == "expanding") & (x.date == df_day.date.max())].port_rtn.iloc[0]
    results.append(_check("Equal weight = mean of leverage * signal_rtn across names",
        np.isclose(eq_day, (df_day.leverage * df_day.signal_rtn).mean())))

    df_stats = runner.get_stats()
    df_sig   = runner.get_stats(level = "signal", vol_targets = ["no_vol_target"])
    results.append(_check("Stats: one row per model x sample x vol target",
        (len(df_stats) == 4 * len(runner.results)) and
        (set(df_stats["sample"]) == {"in_sample", "train", "test", "expanding"})))
    results.append(_check("Signal stats: one row per name x model x sample",
        len(df_sig) == len(runner.signal_data.groupby(["ticker", "model", "sample"]))))

    # 12. custom group / date columns, apply before run raises
    df_alt     = df.rename(columns = {"ticker": "signal_name", "date": "Date"})
    fitter_alt = QuantileFitter(rtn_col = "px_rtn", group_col = "signal_name", date_col = "Date")
    runner_alt = StrategyRunner(fitter_alt, "signal", min_train_obs = 0).run(df_alt)
    df_alt_ewm = runner_alt.apply(EWMVolTarget(span = 100))

    raised = False
    try:
        StrategyRunner(fitter, "signal").apply(NoVolTarget())
    except ValueError:
        raised = True

    results.append(_check("Custom group / date cols match, apply before run raises",
        (list(df_alt_ewm.columns[:2]) == ["Date", "signal_name"]) and
        np.allclose(df_alt_ewm.leverage.values, df_ewm.leverage.values, equal_nan = True) and raised))

    print("\n{} / {} tests passed".format(sum(results), len(results)))
    print("\nTimings")
    print("run, 3 models ({} names, {:,} rows) : {:.3f}s".format(df.ticker.nunique(), len(runner.signal_data), run_time))
    print("apply one vol target                : {:.3f}s".format(apply_time))

if __name__ == "__main__": main()
