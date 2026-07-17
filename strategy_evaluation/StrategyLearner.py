""""""
"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

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
import random

import pandas as pd
import numpy as np
import util as ut

import marketsimcode as ms
import RTLearner as rt
import BagLearner as bl
import indicators as ind


class StrategyLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """
    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

        self.N = 10     
        self.threshold = 0.02
        self.learner = bl.BagLearner(
            learner=rt.RTLearner,
            kwargs={"leaf_size": 10},
            bags=25,
            boost=False,
        )

    def get_indicators(self, symbol, sd, ed):
        """Same indicator calls as ManualStrategy, raw values (not votes)."""
        bb = ind.bollinger_bands_indicator(symbol, sd, ed)
        macd = ind.macd_indicator(symbol, sd, ed)
        cci = ind.cci_indicator(symbol, sd, ed)
        df = pd.concat([bb, macd, cci], axis=1)
        df.columns = ["bb", "macd", "cci"]
        return df



    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=100000,
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """

        indicators = self.get_indicators(symbol, sd, ed)
        prices = ut.get_data([symbol], pd.date_range(sd, ed))[symbol]  # or however your indicators.py fetches adj close

        future_ret = prices.shift(-self.N) / prices - 1.0

        Y = pd.Series(0, index=prices.index)
        Y[future_ret > (self.threshold + self.impact)] = 1
        Y[future_ret < -(self.threshold + self.impact)] = -1

        data = pd.concat([indicators, Y.rename("Y")], axis=1).dropna()

        data_x = data.iloc[:, :-1].values
        data_y = data["Y"].values

        self.learner.add_evidence(data_x, data_y)

    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=100000,
    ):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """
        
        indicators = self.get_indicators(symbol, sd, ed)
        indicators = indicators.fillna(0)

        preds = self.learner.query(indicators.values)
        preds = np.round(preds)
        preds = np.clip(preds, -1, 1)

        dates = indicators.index
        holdings = 0
        trades = pd.Series(0, index=dates)

        for i, date in enumerate(dates):
            signal = preds[i]
            if signal == 1:
                desired_position = 1000
            elif signal == -1:
                desired_position = -1000
            else:
                desired_position = holdings

            trades[date] = desired_position - holdings
            holdings = desired_position

        self.symbol = symbol
        self.sv = sv
        self.trades = trades
        return trades.to_frame(name=symbol)

    def get_port_val(self):
        portvals = ms.compute_portvals(
            self.trades.to_frame(name=self.symbol),
            symbol=self.symbol,
            start_val=self.sv,
            commission=self.commission,
            impact=self.impact,
        )
        benchmark_trades = self.trades.copy() * 0
        benchmark_trades.iloc[0] = 1000
        benchmark = ms.compute_portvals(
            benchmark_trades.to_frame(name=self.symbol),
            symbol=self.symbol,
            start_val=self.sv,
            commission=self.commission,
            impact=self.impact,
        )
        return portvals, benchmark
    
    def author(self):
        return "aperez374"
    
    def study_group(self):
        return "aperez374"


if __name__ == "__main__":
    print("One does not simply think up a strategy")
