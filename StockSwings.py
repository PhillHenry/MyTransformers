"""Plots the biggest percentage swings in the `close` column, with context either side.

Reads the same `timestamp,open,high,low,close,volume` CSV as StockCnn.py and draws a
grid of one subplot per swing: the close price for `margin` bars before and after the
move, with the swing itself picked out. Handy for eyeballing whether the moves a model
is being asked to predict look like anything at all, or just like noise.

A swing here is one bar's close-to-close change *within a single calendar day*. The
move across an overnight gap is ignored: it is usually the largest change in the file
and has nothing to do with the intraday behaviour being looked for. The context drawn
either side is clipped to the same day for the same reason. The top k are taken by size
of move regardless of direction, and no two are allowed within `margin` bars of each
other -- otherwise one violent minute fills every subplot with the same picture.

    python StockSwings.py --csv prices.csv --top-k 9 --margin 30
    python StockSwings.py --csv prices.csv --top-k 4 --margin 120 --out swings.png

With no --csv it plots the synthetic random walk StockCnn.py demos against.
"""
import argparse
import datetime as dt
from typing import NamedTuple

import matplotlib
import numpy as np

from StockCnn import calendar_columns, read_csv, synthetic_csv


def interactive_backend() -> bool:
    """Whether matplotlib ended up with a backend that can actually open a window.

    It falls back to Agg without complaint when no GUI toolkit will import, and only
    says so as a warning once a figure is shown.
    """
    from matplotlib.backends import BackendFilter, backend_registry
    import matplotlib.pyplot as plt

    return plt.get_backend().lower() in backend_registry.list_builtin(BackendFilter.INTERACTIVE)


class Swing(NamedTuple):
    """One close-to-close move: where it happened, how big it was, and when."""
    index: int                                               # row the move ends on
    change: float                                            # fraction, signed
    timestamp: dt.datetime


class StockSwings:
    """The sharpest moves in a price series, and the bars surrounding each one."""

    def __init__(self, timestamps, closes):
        self.timestamps = list(timestamps)
        self.closes = np.asarray(closes, dtype=np.float64)
        self.days, _ = calendar_columns(self.timestamps)     # also settles any mixed time zones
        # Change into row i from row i-1, so index 0 has no change to speak of.
        self.changes = np.concatenate([[0.0], np.diff(self.closes) / self.closes[:-1]])
        # Row i's change only counts when row i-1 is the same day: an overnight gap is
        # not a swing, it is the market reopening somewhere else.
        self.same_day = np.concatenate([[False], self.days[1:] == self.days[:-1]])

    @classmethod
    def from_csv(cls, path: str):
        timestamps, prices = read_csv(path)
        return cls(timestamps, prices[:, 3])                 # close is the fourth column

    def day_bounds(self, index: int):
        """First and last-plus-one row of the calendar day that row `index` belongs to."""
        day = self.days[index]                               # rows are sorted, so the day is contiguous
        return int(np.searchsorted(self.days, day, "left")), int(np.searchsorted(self.days, day, "right"))

    def top(self, k: int, margin: int):
        """The k largest moves by magnitude, none within `margin` bars of a bigger one.

        Without that spacing rule a single spike -- which moves the price on the way in
        and again on the way out -- would take several of the k slots and every subplot
        would show the same stretch of the series.
        """
        swings, taken = [], []
        for index in np.argsort(-np.abs(np.where(self.same_day, self.changes, 0.0))):
            if not self.same_day[index] or self.changes[index] == 0.0:
                continue
            if any(abs(index - other) <= margin for other in taken):
                continue
            swings.append(Swing(int(index), float(self.changes[index]), self.timestamps[index]))
            taken.append(index)
            if len(swings) == k:
                break
        return swings

    def plot(self, k: int, margin: int, out: str = None):
        """A grid of one subplot per swing; saved to `out`, or shown if that is None."""
        import matplotlib.pyplot as plt

        swings = self.top(k, margin)
        if not swings:
            raise SystemExit("nothing to plot: no two consecutive rows share a calendar day "
                             "and move the close" if not self.same_day.any() else
                             "nothing to plot: the close never moves within a day")
        columns = min(3, len(swings))
        rows = -(-len(swings) // columns)                    # ceiling division
        figure, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 3.6 * rows), squeeze=False)

        for axis, swing in zip(axes.flat, swings):
            # Context stops at the day's edges, so no subplot shows an overnight jump.
            day_start, day_stop = self.day_bounds(swing.index)
            start = max(swing.index - margin, day_start)
            stop = min(swing.index + margin + 1, day_stop)
            window = self.closes[start:stop]
            # Bars are numbered relative to the swing, so every subplot is centred on 0
            # even where the file runs out before the margin does.
            offsets = np.arange(start, stop) - swing.index

            axis.plot(offsets, window, color="#3b6ea5", linewidth=1.1)
            axis.plot([-1, 0], self.closes[swing.index - 1:swing.index + 1],
                      color="#c0392b" if swing.change < 0 else "#1e8449", linewidth=2.4)
            axis.axvline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=0)
            axis.set_title(f"{swing.change:+.2%} at {swing.timestamp:%Y-%m-%d %H:%M}", fontsize=10)
            axis.set_xlabel("bars from the swing")
            axis.set_ylabel("close")
            axis.grid(alpha=0.25, linewidth=0.5)

        for axis in axes.flat[len(swings):]:                 # a partly filled last row
            axis.axis("off")
        figure.suptitle(f"top {len(swings)} same-day close-to-close swings, up to {margin} bars either side")
        figure.tight_layout()

        if out is None and not interactive_backend():
            # Nothing can open a window here -- plt.show() would warn and draw nothing --
            # so write the figure out rather than exiting with an empty screen.
            out = "swings.png"
            print(f"the {plt.get_backend()} backend cannot open a window, so the figure goes to a file")
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
    parser.add_argument("--out", help="write the figure here instead of opening a window "
                                      "(the default when no GUI backend is available)")
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
    within_day = np.abs(swings.changes[swings.same_day])
    print(f"{len(swings.closes)} rows over {len(np.unique(swings.days))} calendar days, "
          f"median absolute same-day move {np.median(within_day) if within_day.size else float('nan'):.3%}")
    for swing in swings.plot(args.top_k, args.margin, args.out):
        print(f"  {swing.timestamp:%Y-%m-%d %H:%M}  {swing.change:+.2%}  row {swing.index}")


if __name__ == "__main__":
    main()
