"""Plots the windows StockCnn.py or StockLstm.py wrote with --hits: open, high, low, close and volume per hit.

Each hit is one prediction -- one the model got right, or with a _negative file one it
got wrong -- with --margin rows of context either side, most confident first. Every hit
gets its own subplot: the four prices as lines, volume as bars on a second axis, and the
bar the prediction was made from (offset 0) marked.
The title is the instrument -- the source file's name up to its first underscore -- with
the timestamp at offset 0 and the model's probability.

    python PlotHits.py hits.csv
    python PlotHits.py hits_aapl.csv,hits_msft.csv --top 6 --margin 30 --out hits.png
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

from StockSwings import interactive_backend

PRICE_COLOURS = {"open": "#8e8e8e", "high": "#1e8449", "low": "#c0392b", "close": "#3b6ea5"}


def read_hits(paths, margin: int, threshold: float = None, top: int = None):
    """Every hit in the files, most confident first, each as the block of rows around it.

    A hits file writes each price row once, with `probability` set only on the rows a
    prediction was made from. A hit's block is put back together here: the rows of its
    source within `margin` of its `row`, on the same day, with `offset` counting rows from
    the hit and the hit's `probability` on every one. `margin` should be no more than the
    model's --horizon, which is all the context the file holds.

    `threshold` leaves out hits less probable than it, and `top` keeps only that many of
    the most probable across all the files -- both before any block is built, as there
    can be tens of thousands.
    """
    candidates = []                                          # (probability, rows of its source, position)
    for path in paths:
        frame = pd.read_csv(path)
        for _, rows in frame.groupby("source", sort=False):
            rows = rows.sort_values("row").reset_index(drop=True)
            # Per source: one file can hold sources with and without an offset in their stamps.
            rows["day"] = pd.to_datetime(rows["timestamp"], utc=True, format="ISO8601").dt.date
            rows["timestamp"] = pd.to_datetime(rows["timestamp"], format="ISO8601")
            probability = rows["probability"]
            hits = np.flatnonzero(probability.notna() & (probability >= (threshold or 0.0)))
            candidates += [(float(probability.iloc[i]), rows, i) for i in hits]
    # Stable, so equally probable hits keep the order of the files and rows they came from.
    candidates.sort(key=lambda candidate: -candidate[0])

    groups = []
    for probability, rows, i in candidates[:top]:
        row, day = rows["row"].iloc[i], rows["day"].iloc[i]
        near = rows.iloc[np.searchsorted(rows["row"], row - margin):np.searchsorted(rows["row"], row + margin + 1)]
        near = near[near["day"] == day]
        groups.append(near.drop(columns="day").assign(offset=near["row"] - row, probability=probability))
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
    parser.add_argument("--margin", type=int, default=60,
                        help="rows either side of each hit to plot; no more than the model's --horizon, "
                             "which is all the context the file holds")
    parser.add_argument("--out", help="write the figure here instead of opening a window "
                                      "(the default when no GUI backend is available)")
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top needs to be at least 1")
    if args.out:
        matplotlib.use("Agg")                                # no display needed to write a file

    groups = read_hits([path.strip() for path in args.hits.split(",") if path.strip()], args.margin, top=args.top)
    if not groups:
        raise SystemExit("nothing to plot: no hits in the file")
    for group in groups:
        print(f"  {title(group)}")
    plot(groups, args.out)


if __name__ == "__main__":
    main()
