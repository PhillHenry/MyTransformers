"""Backtests the predictions StockCnn.py or StockLstm.py wrote with --hits.

Every hit whose probability is at least --threshold is a trade: buy at the close of the
bar the prediction was made from (offset 0), then over the next --horizon bars

  * sell at close * (1 + target) on the first bar whose high reaches it -- or at that
    bar's open, if it opened above the target already, since a resting sell order
    would have filled there -- or else
  * sell at the average close over those bars, if the target was never reached.

Pass both files --hits writes, the correct predictions and the wrong ones:

    python Backtesting.py hits.csv,hits_negative.csv --threshold 0.9 --horizon 60
    python Backtesting.py hits.csv,hits_negative.csv --threshold 0.9 --horizon 60 --trades trades.csv

Between them the two files hold every window of the test split, so the trades here are
every signal the model gave at --threshold -- less those that came while a position in the
same instrument was still open, which are passed over. Trades never overlap on one
instrument, so the combined line is what the strategy would have made.
"""
import argparse
import os
from typing import NamedTuple

import numpy as np
import pandas as pd

from PlotHits import read_hits


class Trade(NamedTuple):
    """One simulated round trip."""
    hits_file: str
    source: str
    timestamp: pd.Timestamp
    probability: float
    entry: float
    exit: float
    exit_reason: str                                         # "target" or "average"
    bars_held: int
    exit_timestamp: pd.Timestamp                             # the bar it sold on, or the horizon's last

    @property
    def gain(self) -> float:
        return self.exit / self.entry - 1.0


class Backtest:
    """Trades every hit at or above `threshold`, exiting at `target` or the `horizon`'s average close.

    Only one position per instrument is open at a time: a signal that comes while the
    last trade on its instrument has yet to sell is passed over, however confident. An
    average-price exit sells across the whole horizon, so that position is open until
    the horizon's last bar. Instruments are independent -- positions in different ones
    can be open together.
    """

    def __init__(self, threshold: float, target: float, horizon: int):
        self.threshold = threshold
        self.target = target
        self.horizon = horizon
        self.trades = []
        self.untradeable = 0                                 # hits with no bar after offset 0
        self.overlapping = 0                                 # signals while a position was open

    def run(self, paths):
        """Every trade from `paths`, in the order they were made, none overlapping another on its instrument."""
        for path in paths:
            for group in read_hits([path], self.horizon, self.threshold):
                self.trade(path, group)
        # Across all the files at once: one instrument's signals are split between them.
        self.trades.sort(key=lambda trade: trade.timestamp)
        kept, sold = [], {}                                  # source -> when its open position sells
        for trade in self.trades:
            if trade.source in sold and trade.timestamp <= sold[trade.source]:
                self.overlapping += 1
                continue
            kept.append(trade)
            sold[trade.source] = trade.exit_timestamp
        self.trades = kept
        return self.trades

    def trade(self, path: str, group: pd.DataFrame):
        entry_row = group[group["offset"] == 0]
        if entry_row.empty:
            self.untradeable += 1
            return
        entry_row = entry_row.iloc[0]
        entry = float(entry_row["close"])
        after = group[(group["offset"] >= 1) & (group["offset"] <= self.horizon)]
        if after.empty:                                      # predicted on the day's last bar
            self.untradeable += 1
            return

        goal = entry * (1.0 + self.target)
        reached = np.flatnonzero(after["high"].to_numpy() >= goal)
        if reached.size:
            bar = after.iloc[reached[0]]
            exit_price, reason = max(goal, float(bar["open"])), "target"
        else:
            bar = after.iloc[-1]
            exit_price, reason = float(after["close"].mean()), "average"
        self.trades.append(Trade(path, entry_row["source"], entry_row["timestamp"],
                                 float(entry_row["probability"]), entry, exit_price, reason,
                                 int(bar["offset"]), bar["timestamp"]))

    @staticmethod
    def summary(trades) -> str:
        """Count, how many reached the target, win rate, and the gains as mean, median, sum and compounded."""
        if not trades:
            return "no trades"
        gains = np.array([trade.gain for trade in trades])
        targets = sum(trade.exit_reason == "target" for trade in trades)
        return (f"{len(trades):4d} trades, {targets} reached the target, {np.mean(gains > 0):.1%} won; "
                f"gain per trade mean {gains.mean():+.3%}, median {np.median(gains):+.3%}; "
                f"summed {gains.sum():+.2%}, compounded {np.prod(1 + gains) - 1:+.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hits", help="comma-separated CSV files written by StockCnn.py or StockLstm.py --hits, "
                                     "normally both the file and its _negative partner")
    parser.add_argument("--threshold", type=float, required=True,
                        help="trade only hits whose probability is at least this")
    parser.add_argument("--target", type=float, default=0.02,
                        help="the gain, as a fraction, at which to sell; use the value the model was trained with")
    parser.add_argument("--horizon", type=int, required=True,
                        help="bars after the buy to wait for the target, and to average over if it never comes; "
                             "the model's --horizon, or less, since that is all the file holds")
    parser.add_argument("--trades", help="write every simulated trade here, as CSV")
    args = parser.parse_args()

    if args.horizon < 1:
        parser.error("--horizon needs to be at least 1")
    paths = [path.strip() for path in args.hits.split(",") if path.strip()]

    backtest = Backtest(args.threshold, args.target, args.horizon)
    trades = backtest.run(paths)
    for trade in trades:
        print(f"{trade.timestamp:%Y-%m-%d %H:%M}  {os.path.basename(trade.source).split('_')[0]:6s} "
              f"p={trade.probability:.3f}  bought {trade.entry:.2f}  sold {trade.exit:.2f} "
              f"({trade.exit_reason}, {trade.bars_held} bars, to {trade.exit_timestamp:%H:%M})  {trade.gain:+.3%}")
    for path in paths:
        print(f"{path}: {Backtest.summary([t for t in trades if t.hits_file == path])}")
    if len(paths) > 1:
        print(f"all: {Backtest.summary(trades)}")
    if trades:
        print(f"average gain over all {len(trades)} trades: {np.mean([t.gain for t in trades]):+.3%}")
    if backtest.overlapping:
        print(f"passed over {backtest.overlapping} signals that came while a position in the same "
              f"instrument was still open")
    if backtest.untradeable:
        print(f"skipped {backtest.untradeable} hits with no bar after the prediction to sell on")

    if args.trades:
        pd.DataFrame([{**trade._asdict(), "gain": trade.gain} for trade in trades]).to_csv(args.trades, index=False)
        print(f"wrote {len(trades)} trades to {args.trades}")


if __name__ == "__main__":
    main()
