from util import get_data, plot_data
import datetime as dt
import pandas as pd
import os
import numpy as np


def author():
    return "aperez374"  

def study_group():
    return "aperez374"

def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
    
    stock = get_data([symbol], pd.date_range(sd, ed))

    #fill missing data
    stock = stock.ffill().bfill()

    stock = stock[symbol]

    #create output dataframe
    result = stock.copy()*0

    current_balance = 0

    for i in range(0, len(stock)-1):
        if stock[i+1] > stock[i]:
            target = 1000
        elif stock[i+1] < stock[i]:
            target = -1000
        else:
            target = current_balance
        
        trade = target - current_balance
        result[i] = trade
        current_balance = target

    result.iloc[-1] = 0
    result = result.to_frame(name="Trades")
    
    return result

