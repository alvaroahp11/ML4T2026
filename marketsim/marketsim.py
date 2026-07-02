""""""
"""MC2-P1: Market simulator.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: aperez374 (replace with your User ID)
GT ID: 904197062 (replace with your GT ID)
"""

import datetime as dt
import os

import numpy as np
import pandas as pd

from util import get_data, plot_data

def author():
    return "aperez374"  

def study_group():
    return "aperez374"  


def compute_portvals(
    orders_file="./orders/orders.csv",
    start_val=1000000,
    commission=9.95,
    impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    #Read in the orders file

    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders = orders.sort_index()
    
    symbols = orders['Symbol'].unique().tolist()

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    start_date = orders.index.min()
    end_date = orders.index.max()
    portvals = get_data(symbols, pd.date_range(start_date, end_date))

    #fill missing data
    portvals = portvals.ffill().bfill()

    #remove days with no trading
    trading_days = portvals.index
    orders = orders[orders.index.isin(trading_days)]

    portvals = portvals[symbols]  # remove SPY
    #rv = pd.DataFrame(index=portvals.index, data=portvals.values)
    portvals["Cash"] = 1
    

    #create the trades dataframe
    trades = portvals.copy(deep=True)
    trades[:] = 0

    for date, row in orders.iterrows():
        symbol = row['Symbol']
        order = row['Order']
        shares = row['Shares']
        stock_value = portvals.at[date,symbol]

        if order == "BUY":
            trades.at[date,symbol]+= shares
            trades.at[date,'Cash']+=stock_value*shares*-1
            trades.at[date, 'Cash'] += -commission - impact * stock_value * shares
        else:
            trades.at[date,symbol] += shares*-1
            trades.at[date,'Cash']+=stock_value*shares
            trades.at[date, 'Cash'] += -commission - impact * stock_value * shares
        

    holdings = trades.copy()
    holdings.at[holdings.index[0], 'Cash'] += start_val
    holdings = holdings.cumsum()
    values = (holdings * portvals).sum(axis=1).to_frame('portval')
    return values

def calculate_statistic_metrics(df):
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


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-01.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = portvals.index.min()
    end_date = portvals.index.max()
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_statistic_metrics(portvals)
    #SPY
    spy = get_data(["SPY"], pd.date_range(start_date, end_date))["SPY"]
    spy = spy.ffill().bfill()
    spy.to_csv("spy.csv")
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = calculate_statistic_metrics(spy)
    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
