import pandas as pd
import numpy as np
import datetime as dt
from util import get_data
import matplotlib.pyplot as plt

def author():
    return "aperez374"  

def study_group():
    return "aperez374"

def normalize(series):
    first_value = series.dropna().iloc[0]
    return series / first_value  

def _bollinger_components(symbol, sd, ed, window=20, num_std_dev=2):

    padded_sd = sd - dt.timedelta(days=window * 2)
    dates = pd.date_range(padded_sd, ed)
    prices = get_data([symbol], dates)[symbol]
    prices = prices.ffill().bfill()

    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    bbp = (prices - lower_band) / (upper_band - lower_band)
    bbp.name = "BBP"

    return {
        "prices": prices.loc[sd:ed],
        "rolling_mean": rolling_mean.loc[sd:ed],
        "upper_band": upper_band.loc[sd:ed],
        "lower_band": lower_band.loc[sd:ed],
        "bbp": bbp.loc[sd:ed],
    }


def bollinger_bands_indicator(symbol, sd, ed, window=20, num_std_dev=2):
    components = _bollinger_components(symbol, sd, ed, window, num_std_dev)
    return components["bbp"]

def _macd_components(symbol, sd, ed, fast=12, slow=26, signal_window=9):
    padded_sd = sd - dt.timedelta(days=slow * 3)
    dates = pd.date_range(padded_sd, ed)
    prices = get_data([symbol], dates)[symbol]
    prices = prices.ffill().bfill()

    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line
    histogram.name = "MACD_Histogram"

    return {
        "prices": prices.loc[sd:ed],
        "ema_fast": ema_fast.loc[sd:ed],
        "ema_slow": ema_slow.loc[sd:ed],
        "macd_line": macd_line.loc[sd:ed],
        "signal_line": signal_line.loc[sd:ed],
        "histogram": histogram.loc[sd:ed],
    }


def macd_indicator(symbol, sd, ed, fast=12, slow=26, signal_window=9):
    components = _macd_components(symbol, sd, ed, fast, slow, signal_window)
    return components["histogram"]


def _stochastic_components(symbol, sd, ed, k_window=14, d_window=3, smooth_k=3):
    padded_sd = sd - dt.timedelta(days=k_window * 3)
    dates = pd.date_range(padded_sd, ed)

    close = get_data([symbol], dates, colname="Close")[symbol]
    adj_close = get_data([symbol], dates, colname="Adj Close")[symbol]
    high = get_data([symbol], dates, colname="High")[symbol]
    low = get_data([symbol], dates, colname="Low")[symbol]



    adj_factor = adj_close / close
    high = (high * adj_factor).ffill().bfill()
    low = (low * adj_factor).ffill().bfill()
    adj_close = adj_close.ffill().bfill()

    low_n = low.rolling(window=k_window).min()
    high_n = high.rolling(window=k_window).max()

    percent_k_raw = 100 * (adj_close - low_n) / (high_n - low_n)
    percent_k = percent_k_raw.rolling(window=smooth_k).mean()  
    percent_d = percent_k.rolling(window=d_window).mean()

    result = percent_k - percent_d
    result.name = "Stochastic_K_minus_D"

    return {
        "prices": adj_close.loc[sd:ed],
        "percent_k": percent_k.loc[sd:ed],
        "percent_d": percent_d.loc[sd:ed],
        "result": result.loc[sd:ed],
    }


def stochastic_indicator(symbol, sd, ed, k_window=14, d_window=3):
    components = _stochastic_components(symbol, sd, ed, k_window, d_window)
    return components["result"]

def _cci_components(symbol, sd, ed, window=20, constant=0.015):
    padded_sd = sd - dt.timedelta(days=window * 3)
    dates = pd.date_range(padded_sd, ed)
    prices = get_data([symbol], dates)[symbol]
    prices = prices.ffill().bfill()

    sma = prices.rolling(window=window).mean()
    mean_deviation = prices.rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cci_values = (prices - sma) / (constant * mean_deviation)
    cci_values.name = "CCI"

    return {
        "prices": prices.loc[sd:ed],
        "sma": sma.loc[sd:ed],
        "cci": cci_values.loc[sd:ed],
    }


def cci_indicator(symbol, sd, ed, window=20, constant=0.015):
    components = _cci_components(symbol, sd, ed, window, constant)
    return components["cci"]


def _ppo_components(symbol, sd, ed, fast=12, slow=26, signal_window=9):
    padded_sd = sd - dt.timedelta(days=slow * 3)
    dates = pd.date_range(padded_sd, ed)
    prices = get_data([symbol], dates)[symbol]
    prices = prices.ffill().bfill()

    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    ppo_line = 100 * (ema_fast - ema_slow) / ema_slow
    signal_line = ppo_line.ewm(span=signal_window, adjust=False).mean()
    histogram = ppo_line - signal_line
    histogram.name = "PPO_Histogram"

    return {
        "prices": prices.loc[sd:ed],
        "ema_fast": ema_fast.loc[sd:ed],
        "ema_slow": ema_slow.loc[sd:ed],
        "ppo_line": ppo_line.loc[sd:ed],
        "signal_line": signal_line.loc[sd:ed],
        "histogram": histogram.loc[sd:ed],
    }


def ppo_indicator(symbol, sd, ed, fast=12, slow=26, signal_window=9):
    components = _ppo_components(symbol, sd, ed, fast, slow, signal_window)
    return components["histogram"]