"""Plots the biggest percentage swings in the `close` column, with context either side.

Reads the same `timestamp,open,high,low,close,volume` CSV as StockCnn.py and draws a
grid of one subplot per swing: the close price for `margin` bars before and after the
move, with the swing itself picked out. Handy for eyeballing whether the moves a model
is being asked to predict look like anything at all, or just like noise.

A swing here is one bar's close-to-close change. The top k are taken by size of move
regardless of direction, and no two are allowed within `margin` bars of each other --
otherwise one violent minute fills every subplot with the same picture.

    python StockSwings.py --csv prices.csv --top-k 9 --margin 30
    python StockSwings.py --csv prices.csv --top-k 4 --margin 120 --out swings.png

With no --csv it plots the synthetic random walk StockCnn.py demos against.
"""
import argparse
import datetime as dt
from typing import NamedTuple

import matplotlib
import numpy as np

from StockCnn import read_csv, synthetic_csv


class Swing(NamedTuple):
    """One close-to-close move: where it happened, how big it was, and when."""
    index: int                                               # row the move ends on
    change: float                                            # fraction, signed
    timestamp: dt.datetime
    overnight: bool                                          # the move spans a calendar day


class StockSwings:
    """The sharpest moves in a price series, and the bars surrounding each one."""

    def __init__(self, timestamps, closes):
        self.timestamps = list(timestamps)
        self.closes = np.asarray(closes, dtype=np.float64)
        # Change into row i from row i-1, so index 0 has no change to speak of.
        self.changes = np.concatenate([[0.0], np.diff(self.closes) / self.closes[:-1]])

    @classmethod
    def from_csv(cls, path: str):
        timestamps, prices = read_csv(path)
        return cls(timestamps, prices[:, 3])                 # close is the fourth column

    def top(self, k: int, margin: int):
        """The k largest moves by magnitude, none within `margin` bars of a bigger one.

        Without that spacing rule a single spike -- which moves the price on the way in
        and again on the way out -- would take several of the k slots and every subplot
        would show the same stretch of the series.
        """
        swings, taken = [], []
        for index in np.argsort(-np.abs(self.changes)):
            if index == 0 or self.changes[index] == 0.0:
                continue
            if any(abs(index - other) <= margin for other in taken):
                continue
            before, after = self.timestamps[index - 1], self.timestamps[index]
            swings.append(Swing(int(index), float(self.changes[index]), after,
                                before.date() != after.date()))
            taken.append(index)
            if len(swings) == k:
                break
        return swings

    def plot(self, k: int, margin: int, out: str = None):
        """A grid of one subplot per swing; saved to `out`, or shown if that is None."""
        import matplotlib.pyplot as plt

        swings = self.top(k, margin)
        if not swings:
            raise SystemExit("no price movement to plot")
        columns = min(3, len(swings))
        rows = -(-len(swings) // columns)                    # ceiling division
        figure, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 3.6 * rows), squeeze=False)

        for axis, swing in zip(axes.flat, swings):
            start = max(swing.index - margin, 0)
            stop = min(swing.index + margin + 1, len(self.closes))
            window = self.closes[start:stop]
            # Bars are numbered relative to the swing, so every subplot is centred on 0
            # even where the file runs out before the margin does.
            offsets = np.arange(start, stop) - swing.index

            axis.plot(offsets, window, color="#3b6ea5", linewidth=1.1)
            axis.plot([-1, 0], self.closes[swing.index - 1:swing.index + 1],
                      color="#c0392b" if swing.change < 0 else "#1e8449", linewidth=2.4)
            axis.axvline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=0)
            axis.set_title(f"{swing.change:+.2%} at {swing.timestamp:%Y-%m-%d %H:%M}"
                           f"{' (overnight)' if swing.overnight else ''}", fontsize=10)
            axis.set_xlabel("bars from the swing")
            axis.set_ylabel("close")
            axis.grid(alpha=0.25, linewidth=0.5)

        for axis in axes.flat[len(swings):]:                 # a partly filled last row
            axis.axis("off")
        figure.suptitle(f"top {len(swings)} close-to-close swings, {margin} bars either side")
        figure.tight_layout()

        if out:
            figure.savefig(out, dpi=140)
            print(f"saved to {out}")
        else:
            plt.show()
        plt.close(figure)
        return swings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="timestamp,open,high,low,close,volume (default: generate synthetic data)")
    parser.add_argument("--top-k", type=int, default=9, help="how many swings to plot")
    parser.add_argument("--margin", type=int, default=30, help="bars of context drawn either side of each swing")
    parser.add_argument("--out", help="write the figure here instead of opening a window")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k needs to be at least 1")
    if args.margin < 1:
        parser.error("--margin needs to be at least 1")
    if args.out:
        matplotlib.use("Agg")                                # no display needed to write a file

    path = args.csv or synthetic_csv("/tmp/synthetic_prices.csv")
    if not args.csv:
        print(f"no --csv given, generated synthetic data at {path}")

    swings = StockSwings.from_csv(path)
    print(f"{len(swings.closes)} rows, median absolute move {np.median(np.abs(swings.changes[1:])):.3%}")
    for swing in swings.plot(args.top_k, args.margin, args.out):
        print(f"  {swing.timestamp:%Y-%m-%d %H:%M}  {swing.change:+.2%}  row {swing.index}"
              f"{'  overnight' if swing.overnight else ''}")


if __name__ == "__main__":
    main()
