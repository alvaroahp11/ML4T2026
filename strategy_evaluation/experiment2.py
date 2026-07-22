import datetime as dt
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import StrategyLearner as sl
import marketsimcode as msc
import indicators as ind


def author():
    return "aperez374"


def study_group():
    return "aperez374"


def _benchmark_cr(symbol, sd, ed, sv, commission, impact):
    idx = ind.bollinger_bands_indicator(symbol, sd, ed).index
    trades = pd.Series(0.0, index=idx)
    trades.iloc[0] = 1000.0
    portvals = msc.compute_portvals(trades.to_frame(symbol), symbol, sv, commission, impact)
    cr, adr, sddr, sr = msc.calculate_statistic_metrics(portvals)
    return cr


def run_experiment2(symbol="JPM", sv=100000, commission=0.0,
                    impact_values=(0.0, 0.005, 0.01, 0.025, 0.05)):
    os.makedirs("images", exist_ok=True)

    sd, ed = dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)

    rows = []
    for impact in impact_values:
        learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
        learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
        trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        portvals, _ = learner.get_port_val()

        cr, adr, sddr, sr = msc.calculate_statistic_metrics(portvals)
        n_trades = int((trades[symbol] != 0).sum())

        rows.append({
            "impact": impact,
            "cumulative_return": cr,
            "num_trades": n_trades,
        })

    results = pd.DataFrame(rows)
    results.to_csv("p8_experiment2_results.csv", index=False)

    bench_cr = _benchmark_cr(symbol, sd, ed, sv, commission, impact_values[0])

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(results["impact"], results["cumulative_return"], marker="o", color="blue",
             label="Strategy Learner")
    ax1.set_xlabel("Impact")
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title(f"Experiment 2: Impact vs Cumulative Return ({symbol}, In-Sample)")
    ax1.legend(loc="best")
    ax1.grid(True)
    fig1.tight_layout()
    fig1.savefig("images/experiment2_cumulative_return.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(results["impact"], results["num_trades"], marker="o", color="green",
             label="Strategy Learner")
    ax2.set_xlabel("Impact")
    ax2.set_ylabel("Number of Trades")
    ax2.set_title(f"Experiment 2: Impact vs Number of Trades ({symbol}, In-Sample)")
    ax2.legend(loc="best")
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig("images/experiment2_num_trades.png")
    plt.close(fig2)

    return results


def main(symbol="JPM", sv=100000, commission=0.0,
         impact_values=(0.0, 0.005, 0.01, 0.025, 0.05)):
    return run_experiment2(symbol=symbol, sv=sv, commission=commission,
                           impact_values=impact_values)


if __name__ == "__main__":
    print(main())