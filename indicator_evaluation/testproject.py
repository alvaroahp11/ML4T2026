import TheoreticallyOptimalStrategy as tos
import datetime as dt
from util import get_data, plot_data
import indicators as ind

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def author():
    return "aperez374"  

def study_group():
    return "aperez374"


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

def plot_results(df1, df2, label1, label2, outdir="./images"):
    plt.plot(df1, label=label1, color="red")
    plt.plot(df2, label=label2, color="purple")
    plt.title("Portfolio Value Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Return")
    plt.legend()
    plt.grid(True, linestyle ="--")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=30, ha="right")

    plt.savefig(f'{outdir}/portfolio_comparison.png')


def plot_bollinger_bands(symbol, sd, ed, window=20, num_std_dev=2, outdir="./images"):

    components = ind._bollinger_components(symbol, sd, ed, window, num_std_dev)

    prices = components["prices"]
    rolling_mean = components["rolling_mean"]
    upper_band = components["upper_band"]
    lower_band = components["lower_band"]
    bbp = components["bbp"]

    norm_prices = prices / prices.iloc[0]
    norm_mean = rolling_mean / prices.iloc[0]
    norm_upper = upper_band / prices.iloc[0]
    norm_lower = lower_band / prices.iloc[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    ax1.fill_between(norm_upper.index, norm_lower, norm_upper,
                      color='grey', alpha=0.2, label='Bollinger Band Range')
    ax1.plot(norm_prices.index, norm_prices, label='Price (normalized)', color='navy')
    ax1.plot(norm_mean.index, norm_mean, label='SMA (20)',
              color='darkorange', linestyle='--')
    ax1.set_title(f'Bollinger Bands for {symbol}')
    ax1.set_ylabel('Normalized Price')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    ax2.plot(bbp.index, bbp, label='BBP', color='purple')
    ax2.axhline(1.0, color='red', linestyle=':', linewidth=1, label='Overbought (1.0)')
    ax2.axhline(0.0, color='green', linestyle=':', linewidth=1, label='Oversold (0.0)')
    ax2.set_ylabel('BBP')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{outdir}/bollinger_bands_{symbol}.png')
    plt.close(fig)

def plot_macd(symbol, sd, ed, fast=12, slow=26, signal_window=9, outdir="./images"):
    c = ind._macd_components(symbol, sd, ed, fast, slow, signal_window)

    base = c["prices"].iloc[0]
    norm_prices = c["prices"] / base
    norm_ema_fast = c["ema_fast"] / base
    norm_ema_slow = c["ema_slow"] / base

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(norm_prices.index, norm_prices, label='Price (Normalized)', color='navy')
    ax1.plot(norm_ema_fast.index, norm_ema_fast, label=f'EMA({fast})',
              color='darkorange', linestyle='--')
    ax1.plot(norm_ema_slow.index, norm_ema_slow, label=f'EMA({slow})',
              color='green', linestyle='--')
    ax1.set_title(f'MACD for {symbol}')
    ax1.set_ylabel('Normalized Price')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    ax2.plot(c["histogram"].index, c["histogram"], label='MACD Histogram', color='purple')
    ax2.axhline(0.0, color='black', linewidth=1, linestyle=':', label='Zero line (Buy/Sell)')
    ax2.set_ylabel('MACD Histogram')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{outdir}/macd_{symbol}.png')
    plt.close(fig)

def plot_stochastic(symbol, sd, ed, k_window=14, d_window=3, smooth_k=3, outdir="./images"):
    c = ind._stochastic_components(symbol, sd, ed, k_window, d_window, smooth_k)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                          gridspec_kw={'height_ratios': [3, 2, 2]})

    # Panel 1: Precio normalizado
    norm_prices = c["prices"] / c["prices"].iloc[0]
    ax1.plot(norm_prices.index, norm_prices, label='Price (Normalized)', color='navy')
    ax1.set_title(f'Stochastic Oscillator for {symbol}')
    ax1.set_ylabel('Normalized Price')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    # Panel 2: %K y %D, escala 0-100, con sus umbrales correctos
    ax2.plot(c["percent_k"].index, c["percent_k"], label='%K', color='darkorange', linestyle='--')
    ax2.plot(c["percent_d"].index, c["percent_d"], label='%D', color='navy', linestyle='--')
    ax2.axhline(80, color='red', linestyle=':', linewidth=1, label='Overbought (80)')
    ax2.axhline(20, color='green', linestyle=':', linewidth=1, label='Oversold (20)')
    ax2.set_ylabel('%K / %D')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)

    # Panel 3: la resta que realmente devuelve el indicador, escala propia centrada en 0
    ax3.plot(c["result"].index, c["result"], label='%K - %D (returned)', color='purple')
    ax3.axhline(0.0, color='black', linewidth=1, linestyle=':', label='Zero line (Buy/Sell)')
    ax3.set_ylabel('%K - %D')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper left')
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{outdir}/stochastic_{symbol}.png')
    plt.close(fig)

def plot_cci(symbol, sd, ed, window=20, constant=0.015, outdir="./images"):
    c = ind._cci_components(symbol, sd, ed, window, constant)

    base = c["prices"].iloc[0]
    norm_prices = c["prices"] / base
    norm_sma = c["sma"] / base

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(norm_prices.index, norm_prices, label='Price (Normalized)', color='navy')
    ax1.plot(norm_sma.index, norm_sma, label=f'SMA({window})',
              color='darkorange', linestyle='--')
    ax1.set_title(f'CCI for {symbol}')
    ax1.set_ylabel('Normalized Price')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    ax2.plot(c["cci"].index, c["cci"], label='CCI (returned)', color='purple')
    ax2.axhline(100, color='red', linestyle=':', linewidth=1, label='Overbought (+100) - Sell')
    ax2.axhline(-100, color='green', linestyle=':', linewidth=1, label='Oversold (-100) - Buy')
    ax2.set_ylabel('CCI')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{outdir}/cci_{symbol}.png')
    plt.close(fig)


def plot_ppo(symbol, sd, ed, fast=12, slow=26, signal_window=9, outdir="./images"):
    c = ind._ppo_components(symbol, sd, ed, fast, slow, signal_window)

    base = c["prices"].iloc[0]
    norm_prices = c["prices"] / base
    norm_ema_fast = c["ema_fast"] / base
    norm_ema_slow = c["ema_slow"] / base

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(norm_prices.index, norm_prices, label='Price (Normalized)', color='navy')
    ax1.plot(norm_ema_fast.index, norm_ema_fast, label=f'EMA({fast})',
              color='darkorange', linestyle='--')
    ax1.plot(norm_ema_slow.index, norm_ema_slow, label=f'EMA({slow})',
              color='green', linestyle='--')
    ax1.set_title(f'PPO for {symbol}')
    ax1.set_ylabel('Normalized Price')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    ax2.plot(c["histogram"].index, c["histogram"], label='PPO Histogram', color='purple')
    ax2.axhline(0.0, color='black', linewidth=1, linestyle=':', label='Zero line (Buy/Sell)')
    ax2.set_ylabel('PPO Histogram (%)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{outdir}/ppo_{symbol}.png')
    plt.close(fig)

if __name__ == "__main__":
    #Create the trades dataframe using the testPolicy function
    df_trades = tos.testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)
    benchmark = df_trades.copy()*0
    benchmark.iloc[0] = 1000

    jpm_portvals = compute_portvals(df_trades, symbol = "JPM", start_val=100000, commission=0, impact=0)
    benchmark_portvals = compute_portvals(benchmark, symbol = "JPM", start_val=100000, commission=0, impact=0)

    #normalized data
    jpm_portvals = jpm_portvals/jpm_portvals.iloc[0]
    benchmark_portvals = benchmark_portvals/benchmark_portvals.iloc[0]

    #plot excercise 1
    plot_results(jpm_portvals, benchmark_portvals, "Optimal Portfolio Return", "Benchmark Return")

    #save port statistics
    cr, adr, sddr, sr = calculate_statistic_metrics(jpm_portvals)
    output = f"Optimal Portfolio: Cumulative Return: {cr}\nAverage Daily Return: {adr}\nStandard Deviation of Daily Returns: {sddr}\nSharpe Ratio: {sr}"
    
    with open("p6_results.txt", "w") as f:
        f.write(output + "\n")

    #indicators
    plot_bollinger_bands(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), window=20, num_std_dev=2, outdir="./images")
    plot_macd(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), fast=12, slow=26, signal_window=9, outdir="./images")
    plot_stochastic(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), k_window=14, d_window=3, outdir="./images")
    plot_cci(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), window=20, constant=0.015, outdir="./images")
    plot_ppo(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), fast=12, slow=26, signal_window=9, outdir="./images")




