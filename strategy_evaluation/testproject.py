import os
import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as msc
import experiment1 as exp1
import experiment2 as exp2
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def author():
    return "aperez374"

def study_group():
    return "aperez374"


def plot_strategy_chart(portvals, benchmark, trades, title):
    portvals = portvals.squeeze()
    benchmark = benchmark.squeeze()

    portvals_norm = portvals / portvals.iloc[0]
    benchmark_norm = benchmark / benchmark.iloc[0]

    if isinstance(trades, pd.DataFrame):
        trade_series = trades.iloc[:, 0].squeeze()
    else:
        trade_series = trades.squeeze()

    long_dates = trade_series[trade_series > 0].index
    short_dates = trade_series[trade_series < 0].index

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(portvals_norm.index, portvals_norm.values, label="Manual Strategy", color="red", linewidth=2)
    ax.plot(benchmark_norm.index, benchmark_norm.values, label="Benchmark", color="purple", linewidth=2)

    for i, d in enumerate(long_dates):
        ax.axvline(d, color="blue", linestyle="--", alpha=0.7, linewidth=1,
                label="Long Entry" if i == 0 else None)

    for i, d in enumerate(short_dates):
        ax.axvline(d, color="black", linestyle="--", alpha=0.7, linewidth=1,
                label="Short Entry" if i == 0 else None)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Value")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()

    os.makedirs("./images", exist_ok=True)
    filename = os.path.join("./images", f"{title.replace(' ', '_')}.png")
    fig.savefig(filename)
    plt.close(fig)


def manual_strategy():
    #in sample
    in_sample = ms.ManualStrategy(impact=0.005, commission=9.95)
    in_sample_trades = in_sample.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    portvals, benchmark = in_sample.get_port_val()
    plot_strategy_chart(portvals, benchmark, in_sample_trades, "In sample Manual Strategy vs benchmark")
    cr, adr, sddr, sr = msc.calculate_statistic_metrics(portvals)
    sr_cr, sr_adr, sr_sddr, sr_sr = msc.calculate_statistic_metrics(benchmark)
    output = f"Manual Strategy - In Sample: \nCumulative Return: {cr:.6f}\nAverage Daily Return: {adr:.6f}\nStandard Deviation of Daily Returns: {sddr:.6f}\nSharpe Ratio: {sr:.6f}"
    benchmark_output = f"Benchmark - In Sample:\nCumulative Return: {sr_cr:.6f}\nAverage Daily Return: {sr_adr:.6f}\nStandard Deviation of Daily Returns: {sr_sddr:.6f}\nSharpe Ratio: {sr_sr:.6f}"


    #out of sample
    out_sample = ms.ManualStrategy(impact=0.005, commission=9.95)
    out_sample_trades = out_sample.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    out_sample_portvals, out_sample_benchmark = out_sample.get_port_val()
    plot_strategy_chart(out_sample_portvals, out_sample_benchmark, out_sample_trades, "Out of sample Manual Strategy vs benchmark")
    cr_os, adr_os, sddr_os, sr_os = msc.calculate_statistic_metrics(out_sample_portvals)
    sr_cr_os, sr_adr_os, sr_sddr_os, sr_sr_os = msc.calculate_statistic_metrics(out_sample_benchmark)
    output_os = f"Manual Strategy - Out of Sample: \nCumulative Return: {cr_os:.6f}\nAverage Daily Return: {adr_os:.6f}\nStandard Deviation of Daily Returns: {sddr_os:.6f}\nSharpe Ratio: {sr_os:.6f}"
    benchmark_output_os = f"Benchmark - Out of Sample: \nCumulative Return: {sr_cr_os:.6f}\nAverage Daily Return: {sr_adr_os:.6f}\nStandard Deviation of Daily Returns: {sr_sddr_os:.6f}\nSharpe Ratio: {sr_sr_os:.6f}"

    with open("p8_manual_results.txt", "w") as f:
        f.write(output + "\n"+ "\n")
        f.write(benchmark_output + "\n"+ "\n")
        f.write(output_os+"\n"+ "\n")
        f.write(benchmark_output_os+"\n"+ "\n")

def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 904197062  # replace with your GT ID number


if __name__ == "__main__":
    np.random.seed(gtid())
    manual_strategy()
    exp1.main()
    exp2.main()