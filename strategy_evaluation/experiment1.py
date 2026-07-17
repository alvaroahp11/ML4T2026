import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as msc


def author():
    return "aperez374"  # replace with your GT Canvas ID


def study_group():
    return "aperez374"  # if you are in a study group


def run_benchmark(symbol, sd, ed, sv=100000, commission=9.95, impact=0.005):
    """Create trades for benchmark: buy 1000 shares on first day and hold."""
    dates = pd.date_range(sd, ed)
    trades = pd.DataFrame(index=dates, columns=[symbol])
    trades.values[:, :] = 0.0
    first_day = trades.index[0]
    trades.loc[first_day, symbol] = 1000.0
    portvals = msc.compute_portvals(
        trades, symbol, start_val=sv, commission=commission, impact=impact
    )
    return portvals


def normalize(series):
    return series / series.iloc[0]


def summarize(name, portvals):
    portvals = portvals.squeeze()
    daily_rets = portvals.pct_change().dropna()
    cr = portvals.iloc[-1] / portvals.iloc[0] - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()

    print(f"{name} results:")
    print(f"  Cumulative return: {cr:.4f}")
    print(f"  Mean daily return: {adr:.6f}")
    print(f"  Std of daily return: {sddr:.6f}")
    print()


def main():
    symbol = "JPM"
    sv = 100000

    # in sample and out of sample dates
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)

    sd_out = dt.datetime(2010, 1, 1)
    ed_out = dt.datetime(2011, 12, 31)

    # ManualStrategy instance
    manual = ms.ManualStrategy(verbose=False)

    # StrategyLearner instance
    learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=0.0)

    # train learner in sample
    learner.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)

    # experiment settings
    commission = 9.95
    impact = 0.005

    # IN SAMPLE RUNS
    trades_manual_in = manual.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    portvals_manual_in = msc.compute_portvals(
        trades_manual_in, symbol, start_val=sv, commission=commission, impact=impact
    )

    trades_learner_in = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    portvals_learner_in = msc.compute_portvals(
        trades_learner_in, symbol, start_val=sv, commission=commission, impact=impact
    )

    portvals_bench_in = run_benchmark(
        symbol, sd_in, ed_in, sv=sv, commission=commission, impact=impact
    )

    # summarize in sample
    summarize("In sample Benchmark", portvals_bench_in)
    summarize("In sample ManualStrategy", portvals_manual_in)
    summarize("In sample StrategyLearner", portvals_learner_in)

    # plot in sample
    plt.figure(figsize=(10, 6))
    norm_bench_in = normalize(portvals_bench_in)
    norm_manual_in = normalize(portvals_manual_in)
    norm_learner_in = normalize(portvals_learner_in)

    plt.plot(norm_bench_in.index, norm_bench_in.values, color="purple", label="Benchmark")
    plt.plot(norm_manual_in.index, norm_manual_in.values, color="red", label="ManualStrategy")
    plt.plot(norm_learner_in.index, norm_learner_in.values, color="blue", label="StrategyLearner")

    plt.title("In sample JPM portfolio values")
    plt.xlabel("Date")
    plt.ylabel("Normalized value")
    plt.legend()
    plt.grid(True)
    plt.savefig("./images/experiment1_in_sample.png", dpi=150)

    # OUT OF SAMPLE RUNS
    trades_manual_out = manual.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)
    portvals_manual_out = msc.compute_portvals(
        trades_manual_out, symbol, start_val=sv, commission=commission, impact=impact
    )

    trades_learner_out = learner.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)
    portvals_learner_out = msc.compute_portvals(
        trades_learner_out, symbol, start_val=sv, commission=commission, impact=impact
    )

    portvals_bench_out = run_benchmark(
        symbol, sd_out, ed_out, sv=sv, commission=commission, impact=impact
    )

    # summarize out of sample
    summarize("Out of sample Benchmark", portvals_bench_out)
    summarize("Out of sample ManualStrategy", portvals_manual_out)
    summarize("Out of sample StrategyLearner", portvals_learner_out)

    # plot out of sample
    plt.figure(figsize=(10, 6))
    norm_bench_out = normalize(portvals_bench_out)
    norm_manual_out = normalize(portvals_manual_out)
    norm_learner_out = normalize(portvals_learner_out)

    plt.plot(norm_bench_out.index, norm_bench_out.values, color="purple", label="Benchmark")
    plt.plot(norm_manual_out.index, norm_manual_out.values, color="red", label="ManualStrategy")
    plt.plot(norm_learner_out.index, norm_learner_out.values, color="blue", label="StrategyLearner")

    plt.title("Out of sample JPM portfolio values")
    plt.xlabel("Date")
    plt.ylabel("Normalized value")
    plt.legend()
    plt.grid(True)
    plt.savefig("./images/experiment1_out_of_sample.png", dpi=150)




if __name__ == "__main__":
    main()