"""Plots the windows StockCnn.py or StockLstm.py wrote with --hits: open, high, low, close and volume per hit.

Each group of rows sharing a `hit` value is one prediction the model got right, with its
context either side. Every group gets its own subplot: the four prices as lines, volume
as bars on a second axis, and the bar the prediction was made from (offset 0) marked.
The title is the instrument -- the source file's name up to its first underscore -- with
the timestamp at offset 0 and the model's probability.

    python PlotHits.py hits.csv
    python PlotHits.py hits_aapl.csv,hits_msft.csv --top 6 --out hits.png
"""
import argparse
import os

import matplotlib
import pandas as pd

from StockSwings import interactive_backend

PRICE_COLOURS = {"open": "#8e8e8e", "high": "#1e8449", "low": "#c0392b", "close": "#3b6ea5"}


def read_hits(paths):
    """Every hits file's groups, in file order and then in the order they were written.

    `hit` numbers restart in each file, so a group is keyed by (file, hit) rather than
    the bare number.
    """
    groups = []
    for path in paths:
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        groups += [group.sort_values("offset") for _, group in frame.groupby("hit", sort=False)]
    return groups


def title(group) -> str:
    """Instrument, when the prediction was made, and how sure the model was."""
    instrument = os.path.basename(group["source"].iloc[0]).split("_")[0]
    centre = group[group["offset"] == 0]
    row = centre.iloc[0] if len(centre) else group.iloc[len(group) // 2]
    return f"{instrument}  {row['timestamp']:%Y-%m-%d %H:%M}  p={row['probability']:.3f}"


def plot(groups, out: str = None):
    """A grid of one subplot per hit; saved to `out`, or shown if that is None."""
    import matplotlib.pyplot as plt

    columns = min(3, len(groups))
    rows = -(-len(groups) // columns)                        # ceiling division
    figure, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 3.8 * rows), squeeze=False)

    for axis, group in zip(axes.flat, groups):
        offsets = group["offset"].to_numpy()
        # Volume goes on its own axis, drawn underneath: its scale has nothing to do with price.
        volume_axis = axis.twinx()
        volume_axis.bar(offsets, group["volume"], width=0.8, color="#b0b0b0", alpha=0.45)
        volume_axis.set_ylabel("volume", color="#777777")
        volume_axis.tick_params(axis="y", labelcolor="#777777", labelsize=8)
        volume_axis.set_ylim(0, group["volume"].max() * 3 or 1)  # keep the bars in the bottom third
        axis.set_zorder(volume_axis.get_zorder() + 1)
        axis.patch.set_visible(False)

        for column, colour in PRICE_COLOURS.items():
            axis.plot(offsets, group[column], color=colour, linewidth=1.8 if column == "close" else 1.0,
                      label=column)
        axis.axvline(0, color="#999999", linewidth=0.8, linestyle="--")
        axis.set_title(title(group), fontsize=10)
        axis.set_xlabel("bars from the prediction")
        axis.set_ylabel("price")
        axis.grid(alpha=0.25, linewidth=0.5)

    axes.flat[0].legend(fontsize=8, loc="upper left")
    for axis in axes.flat[len(groups):]:                     # a partly filled last row
        axis.axis("off")
    figure.tight_layout()

    if out is None and not interactive_backend():
        out = "hits.png"
        print(f"the {plt.get_backend()} backend cannot open a window, so the figure goes to a file")
    if out:
        figure.savefig(out, dpi=140)
        print(f"saved to {out}")
    else:
        plt.show()
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hits", help="comma-separated CSV files written by StockCnn.py or StockLstm.py --hits")
    parser.add_argument("--top", type=int, default=12, help="how many hits to plot")
    parser.add_argument("--out", help="write the figure here instead of opening a window "
                                      "(the default when no GUI backend is available)")
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top needs to be at least 1")
    if args.out:
        matplotlib.use("Agg")                                # no display needed to write a file

    groups = read_hits([path.strip() for path in args.hits.split(",") if path.strip()])[:args.top]
    if not groups:
        raise SystemExit("nothing to plot: no hits in the file")
    for group in groups:
        print(f"  {title(group)}")
    plot(groups, args.out)


if __name__ == "__main__":
    main()
