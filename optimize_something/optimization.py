""""""
"""MC1-P2: Optimize a portfolio.

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

Student Name: Alvaro Andres Henriquez Perez
GT User ID: aperez374
GT ID: 904197062
"""


import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from util import get_data, plot_data
import scipy.optimize as spo

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "aperez374"  # replace tb34 with your Georgia Tech username.

def study_group():
    return "aperez374"

def fill_missing_data(df):
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df

def normalized_data(df):
    return df/df.iloc[0]

def calculate_allocation(df):

    n = df.shape[1]
    guess = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = ({'type': 'eq', 'fun': lambda a: np.sum(a) - 1.0},)

    result = spo.minimize(f, guess, args=(df,), method='SLSQP',bounds=bounds, constraints=constraints)
    return result.x

def f(alloc, df):
    df = df * alloc
    df = df.sum(axis=1)
    cr, adr, sddr, sr = calculate_statistic_metrics(df)
    return sr*-1

def plot_df(stocks, spy):
    plt.plot(stocks, label="Portfolio", color="blue")
    plt.plot(spy, label="SPY", color="green")
    plt.legend()
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.title("Daily Portfolio Value and SPY")
    plt.grid(True, linestyle ="--")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("./images/figure1.png")

def calculate_statistic_metrics(df):
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

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 1, 1),
    syms=["GOOG", "AAPL", "GLD", "XOM"],
    gen_plot=False,
):
    """
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and
    statistics.

    :param sd: A datetime object that represents the start date, defaults to 1/1/2008
    :type sd: datetime
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009
    :type ed: datetime
    :param syms: A list of symbols that make up the portfolio (note that your code should support any
        symbol in the data directory)
    :type syms: list
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your
        code with gen_plot = False.
    :type gen_plot: bool
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,
        standard deviation of daily returns, and Sharpe ratio
    :rtype: tuple
    """

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY

    # ffill missing data
    prices_all = fill_missing_data(prices_all)
    # normalized data
    prices_all = normalized_data(prices_all)

    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later


    #calculate allocation and the portfolio value
    alloc = calculate_allocation(prices)

    #calculate portfolio based on the allocations
    prices = prices * alloc
    prices = prices.sum(axis=1)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        plot_df(prices, prices_SPY)

    #now that the graph is done, we need to calculate the statistics of the portfolio
    cr, adr, sddr, sr = calculate_statistic_metrics(prices) # add code here to compute stats
    return alloc, cr, adr, sddr, sr


def test_code():
    """
    This function WILL NOT be called by the auto grader.
    """

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=True)

    # Print statistics
    #print(f"Start Date: {start_date}")
    #print(f"End Date: {end_date}")
    #print(f"Symbols: {symbols}")
    #print(f"Allocations:{allocations}")
    #print(f"Sharpe Ratio: {sr}")
    #print(f"Volatility (stdev of daily returns): {sddr}")
    #print(f"Average Daily Return: {adr}")
    #print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
