import datetime as dt
import os

import numpy as np
import pandas as pd

from util import get_data, plot_data


def compute_portvals(df_trades, symbol, start_val=100000, commission=9.95, impact=0.005):

    df_trades = df_trades.sort_index()

    start_date = df_trades.index.min()
    end_date = df_trades.index.max()

    # get prices for just this symbol over the date range
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices.ffill().bfill()
    prices = prices[[symbol]]  # drop SPY, keep as DataFrame

    # restrict trades to actual trading days (in case df_trades has extra dates)
    df_trades = df_trades[df_trades.index.isin(prices.index)]

    # cash column alongside the symbol column
    prices["Cash"] = 1.0

    trades = prices.copy(deep=True)
    trades[:] = 0.0

    for date, trade_row in df_trades.iterrows():

        shares = trade_row.iloc[0]  # the single trade value for that day
        if shares == 0:
            continue

        stock_price = prices.at[date, symbol]

        trades.at[date, symbol] += shares
        trades.at[date, "Cash"] += -shares * stock_price
        # commission + impact charged any time a nonzero trade happens
        trades.at[date, "Cash"] += -commission - impact * abs(shares) * stock_price

    holdings = trades.copy()
    holdings.at[holdings.index[0], "Cash"] += start_val
    holdings = holdings.cumsum()

    portvals = (holdings * prices).sum(axis=1).to_frame("portval")
    return portvals

def calculate_statistic_metrics(df):
    df = df.squeeze()
    df = df/df.iloc[0]
    daily_return = (df[1:]/df[:-1].values) - 1
    #cumulative return
    cr = (df.iloc[-1]/df.iloc[0])-1
    #average return
    adr = daily_return.mean()
    #standard deviation
    sddr = daily_return.std()

    #sharpe ratio
    #the real calculation is square root of 252 (because our sample is daily it should 252, it could be weekly 52, or monthly 12) * mean(daily_return-daily_riskfree)/std(daily_return). the daily risk free its calculated root 256 of 1*%of the bank example 1*0.1

    sr = (252**(1/2))*(adr/sddr)
    return cr, adr, sddr, sr