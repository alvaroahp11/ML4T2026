import datetime as dt
import indicators as ind
import pandas as pd
import marketsimcode as ms


class ManualStrategy:
    
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def author(self):
        return "aperez374"
    
    def study_group(self):
        return "aperez374"


    def add_evidence(self, symbol="IBM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=100000):
        pass
    

    def testPolicy(self, symbol="IBM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=100000):
        bb = ind.bollinger_bands_indicator(symbol, sd, ed)
        macd = ind.macd_indicator(symbol, sd, ed)
        cci = ind.cci_indicator(symbol, sd, ed)

        dates = bb.index
        bb_vote = pd.Series(0, index=dates)
        macd_vote = pd.Series(0, index=dates)
        cci_vote = pd.Series(0, index=dates)
        
        #bb_operations
        bb_vote[bb>1] = -1
        bb_vote[bb<0] = 1
        #cci operation
        cci_vote[cci>100] = -1
        cci_vote[cci<-100] = 1
        #macd operation
        macd_vote[macd>0.5] = -1
        macd_vote[macd<-0.5] = 1

        #final vote
        final_vote = bb_vote + macd_vote + cci_vote
        target = pd.Series(0, index=dates)
        target[final_vote >= 2] = 1
        target[final_vote <= -2] = -1


        holdings = 0
        trades = pd.Series(0, index=dates)

        for date in dates:
            signal = target[date]
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

if __name__ == "__main__":
    print("ManualStrategy.py")

