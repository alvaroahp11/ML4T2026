import datetime as dt
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import ManualStrategy as ms_strategy
import StrategyLearner as sl
import marketsimcode as msc
import indicators as ind


def author():
    return "aperez374"


def study_group():
    return "aperez374"


def _benchmark_portvals(symbol, sd, ed, sv, commission, impact):
    idx = ind.bollinger_bands_indicator(symbol, sd, ed).index
    trades = pd.Series(0.0, index=idx)
    trades.iloc[0] = 1000.0
    return msc.compute_portvals(trades.to_frame(symbol), symbol, sv, commission, impact)


def _normalize(portvals):
    series = portvals.squeeze()
    return series / series.iloc[0]


def _plot(manual_norm, learner_norm, bench_norm, label, symbol):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(manual_norm.index, manual_norm.values, label="Manual Strategy", color="red")
    ax.plot(learner_norm.index, learner_norm.values, label="Strategy Learner", color="green")
    ax.plot(bench_norm.index, bench_norm.values, label="Benchmark", color="purple")
    ax.set_title(f"Experiment 1: {label} ({symbol})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio Value")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fname = f"images/experiment1_{label.lower().replace(' ', '_')}.png"
    fig.savefig(fname)
    plt.close(fig)


def run_experiment1(symbol="JPM", sv=100000, impact=0.005, commission=9.95):
    os.makedirs("images", exist_ok=True)

    in_sd, in_ed = dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)
    out_sd, out_ed = dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)

    manual_in = ms_strategy.ManualStrategy(impact=impact, commission=commission)
    manual_in.testPolicy(symbol=symbol, sd=in_sd, ed=in_ed, sv=sv)
    manual_in_portvals, _ = manual_in.get_port_val()

    manual_out = ms_strategy.ManualStrategy(impact=impact, commission=commission)
    manual_out.testPolicy(symbol=symbol, sd=out_sd, ed=out_ed, sv=sv)
    manual_out_portvals, _ = manual_out.get_port_val()

    learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
    learner.add_evidence(symbol=symbol, sd=in_sd, ed=in_ed, sv=sv)

    learner.testPolicy(symbol=symbol, sd=in_sd, ed=in_ed, sv=sv)
    learner_in_portvals, _ = learner.get_port_val()

    learner.testPolicy(symbol=symbol, sd=out_sd, ed=out_ed, sv=sv)
    learner_out_portvals, _ = learner.get_port_val()

    bench_in = _benchmark_portvals(symbol, in_sd, in_ed, sv, commission, impact)
    bench_out = _benchmark_portvals(symbol, out_sd, out_ed, sv, commission, impact)

    _plot(_normalize(manual_in_portvals), _normalize(learner_in_portvals),
          _normalize(bench_in), "In Sample", symbol)
    _plot(_normalize(manual_out_portvals), _normalize(learner_out_portvals),
          _normalize(bench_out), "Out of Sample", symbol)

    rows = {}
    for label, pv in [("Manual In", manual_in_portvals), ("Learner In", learner_in_portvals),
                       ("Benchmark In", bench_in), ("Manual Out", manual_out_portvals),
                       ("Learner Out", learner_out_portvals), ("Benchmark Out", bench_out)]:
        cr, adr, sddr, sr = msc.calculate_statistic_metrics(pv)
        rows[label] = {"cum_return": cr, "mean_daily_return": adr, "stdev_daily_return": sddr}

    summary = pd.DataFrame(rows).T.round(6)
    summary.to_csv("p8_experiment1_summary.csv")
    return summary


def main(symbol="JPM", sv=100000, impact=0.005, commission=9.95):
    return run_experiment1(symbol=symbol, sv=sv, impact=impact, commission=commission)


if __name__ == "__main__":
    print(main())