"""A 1D CNN that flags bars from which price is likely to rise 2% within the next 20 rows.

Reads a CSV of `timestamp,open,high,low,close,volume`, turns it into overlapping
windows of scale-free features, and trains a convolutional classifier on the binary
label "did the high reach close * 1.02 within the next H rows *and* before midnight?".

Only bars timestamped between 13:35 and 19:55 UTC inclusive are used; anything outside
that window is dropped as the file is read, so it contributes to neither features nor
labels.

Finer bars are merged into 1-minute bars as they are read, so a row is a minute in every
file and `--window` is a number of minutes.

Nothing is ever compared across working days -- bar the optional N-day moving averages of
the close, which only look back at finished days. Every window lies wholly inside one UTC
day, the label's look-ahead stops at that day's last bar, and the backward-looking
features -- the return and the volume change -- are never taken from yesterday's close
against today's open. A day is therefore self-contained: the overnight gap is not a
signal the model can see or learn.

The same-calendar-day constraint means the effective horizon shrinks as the session
runs down, so the timestamp itself becomes predictive and is fed in as a feature.

Run without arguments for a synthetic random-walk demo:
    python StockCnn.py
Or against real data:
    python StockCnn.py --csv prices.csv
Several files train (and are evaluated as) one model -- the features are scale-free, so
bars from different instruments are comparable:
    python StockCnn.py --csv aapl.csv,msft.csv,nvda.csv
Or a real example:
    python StockCnn.py --csv /home/henryp/Downloads/aapl_dataset_London-Strategic-Edge.csv --target 0.02 --horizon 60
"""
import argparse
import csv
import datetime as dt
import math
import os
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

FEATURE_NAMES = ["log_return", "high_vs_close", "low_vs_close", "open_vs_close", "log_volume_change",
                 "time_of_day", "bars_elapsed_today"]
DEFAULT_MOVING_AVERAGES = (50, 100)


def feature_names(moving_averages=DEFAULT_MOVING_AVERAGES):
    """The fixed features followed by one close-versus-average channel per moving-average window."""
    return FEATURE_NAMES + [f"close_vs_ma{window}" for window in moving_averages]

# The only bars the model ever sees, inclusive of both ends. Everything outside is
# dropped as it is read, so the session boundaries look to the rest of the code exactly
# like the start and end of a day's file.
SESSION_START = dt.time(13, 35)
SESSION_END = dt.time(19, 55)


def parse_timestamp(raw: str) -> dt.datetime:
    """ISO-8601 (with or without a `T`, `Z` or offset) or an epoch number in seconds/millis."""
    text = raw.strip()
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    number = float(text)
    return dt.datetime.fromtimestamp(number / 1000.0 if number > 1e11 else number, tz=dt.timezone.utc)


def as_utc(timestamp: dt.datetime) -> dt.datetime:
    """The same instant in UTC; a naive stamp is taken to be UTC already."""
    return timestamp.astimezone(dt.timezone.utc) if timestamp.tzinfo is not None else timestamp


def in_session(timestamp: dt.datetime) -> bool:
    """Does this bar fall inside the 13:35-19:55 UTC window?

    Offset-aware stamps are converted; naive ones are taken to be UTC already, since
    there is nothing else to go on.
    """
    return SESSION_START <= as_utc(timestamp).time() <= SESSION_END


def read_csv(path: str):
    """The session's rows, sorted by timestamp: parsed timestamps and OHLCV columns as floats.

    The timestamps are no longer just for ordering — the calendar day decides where
    each label's look-ahead window is cut off — and bars outside 13:35-19:55 UTC are
    discarded here, so no feature or label is ever computed from one.
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    stamped = sorted(((parse_timestamp(row["timestamp"]), row) for row in rows), key=lambda pair: pair[0])
    stamped = [pair for pair in stamped if in_session(pair[0])]
    timestamps = [timestamp for timestamp, _ in stamped]
    prices = np.array([[float(row[c]) for c in ("open", "high", "low", "close", "volume")]
                       for _, row in stamped], dtype=np.float64).reshape(-1, 5)
    return to_minute_bars(timestamps, prices)


def to_minute_bars(timestamps, prices):
    """Merge bars that share a clock minute into one 1-minute bar, stamped with that minute.

    Files come at different resolutions -- some in 1-minute bars, some in 15-second
    ones -- and one model needs one: after this a row is a minute everywhere, so a
    window of N rows is N minutes whatever file it came from. Already-1-minute files
    pass through unchanged. Minutes with no bar at all stay missing; `windows` skips them.
    """
    if not timestamps:
        return timestamps, prices
    minutes = [t.replace(second=0, microsecond=0) for t in timestamps]
    starts = np.flatnonzero([True] + [a != b for a, b in zip(minutes[1:], minutes[:-1])])
    ends = np.append(starts[1:], len(minutes)) - 1
    merged = np.stack([prices[starts, 0],
                       np.maximum.reduceat(prices[:, 1], starts),
                       np.minimum.reduceat(prices[:, 2], starts),
                       prices[ends, 3],
                       np.add.reduceat(prices[:, 4], starts)], axis=1)
    return [minutes[i] for i in starts], merged


def calendar_columns(timestamps):
    """Day number (for grouping) and time of day as a fraction, per row.

    `minute_number` turns the pair into an absolute count of minutes, for measuring
    how much time a run of rows spans.

    Both are taken in UTC, which is the zone the session itself is defined in: a local
    date would cut one 13:35-19:55 session into two "days" wherever the offset puts its
    two ends on either side of local midnight, and mixed zones in one file would scatter
    bars across days that never traded together.
    """
    timestamps = [as_utc(t) for t in timestamps]
    days = np.array([t.date().toordinal() for t in timestamps], dtype=np.int64)
    seconds = np.array([t.hour * 3600 + t.minute * 60 + t.second for t in timestamps], dtype=np.float64)
    return days, seconds / 86400.0


def minute_number(days, time_of_day):
    """Minutes since the start of the proleptic calendar, per row, from `calendar_columns`."""
    return days * 1440 + np.rint(time_of_day * 1440).astype(np.int64)


def bars_into_day(days):
    """How many bars each row sits after the first bar of its own day.

    Zero marks a day's opening bar, which is the one row whose backward-looking
    features would otherwise reach into the previous day.
    """
    _, first_of_day = np.unique(days, return_index=True)
    return np.arange(len(days)) - np.repeat(first_of_day, np.diff(np.append(first_of_day, len(days))))


def daily_moving_average(close, days, window: int):
    """Per row, the mean of the last `window` days' closing prices, *not* counting the row's own day.

    A day's close is its last in-session bar, and today's hasn't happened yet, so only
    finished days go into the average: a bar at 14:00 can't know where 19:55 will be.
    Days are the ones present in the data, i.e. trading days. Rows without `window`
    finished days behind them get NaN, not an average over fewer days -- that would put
    short, noisy averages at the start of every file, which is exactly the training split.
    """
    _, first_of_day, day_index = np.unique(days, return_index=True, return_inverse=True)
    last_of_day = np.append(first_of_day[1:], len(days)) - 1
    daily_close = close[last_of_day]
    cumulative = np.concatenate([[0.0], np.cumsum(daily_close)])
    # Entry k is the mean of daily closes k-window .. k-1, for every day k with that many before it.
    averages = np.full(len(daily_close), np.nan)
    averages[window:] = (cumulative[window:-1] - cumulative[:-window - 1]) / window
    return averages[day_index]


def features_and_labels(prices, days, time_of_day, horizon: int, target: float,
                        moving_averages=DEFAULT_MOVING_AVERAGES):
    """Per-row features plus the forward-looking label, aligned on the same index.

    Price features are ratios/differences so the net never sees the absolute price
    level: a model trained on a $10 stock should still work when it trades at $200.
    Row 0 is dropped because the return features need a previous row.

    Every comparison stays inside one working day. The two backward-looking features --
    the return and the volume change -- are diffs against the previous row, so on a day's
    opening bar they would measure today against yesterday's close: an overnight gap, not
    an intraday move. Those rows are neutralised here and their windows are dropped in
    `windows`, so nothing the model sees crosses a day boundary.

    The label for row i is 1 when some bar in rows i+1 .. i+horizon *that falls on the
    same calendar day as row i* trades at or above close[i] * (1 + target). Rows near
    the close therefore have fewer chances, which is why the clock is a feature: with
    no notion of time the model could only average over "how much day is left".

    Each of `moving_averages` adds a channel: how far the close sits above or below the
    moving average of that many previous days' closes, as a fraction, so it too is
    scale-free. These are the one deliberate exception to staying inside a day -- a
    multi-day trend is the point of them -- but they only look back at finished days.
    Rows too early in the file to have the longest average get no label, and so no window.
    """
    open_, high, low, close, volume = (prices[:, i] for i in range(5))

    # True where the diff below spans midnight: row i-1 is yesterday, row i is today.
    overnight = days[1:] != days[:-1]
    # Computed on every row, the first included, so row 0 can still close its day.
    close_vs_averages = [close[1:] / daily_moving_average(close, days, n)[1:] - 1.0
                         for n in moving_averages]
    days, time_of_day = days[1:], time_of_day[1:]

    # Zero is a placeholder, not a measurement: `windows` never lets these rows reach the
    # model, and an overnight gap left in place would be the largest "return" in the file.
    log_return = np.where(overnight, 0.0, np.diff(np.log(close)))
    log_volume_change = np.where(overnight, 0.0, np.diff(np.log1p(volume)))
    # The rest compare a bar with itself, so they are intraday whatever day they fall on.
    high_vs_close = (high[1:] / close[1:]) - 1.0
    low_vs_close = (low[1:] / close[1:]) - 1.0
    open_vs_close = (open_[1:] / close[1:]) - 1.0

    # How far into its day each bar is, counted in bars: the clock as the data actually
    # samples it, which is what limits how many chances the 2% move has left.
    bars_elapsed = bars_into_day(days)

    features = np.stack([log_return, high_vs_close, low_vs_close, open_vs_close, log_volume_change,
                         time_of_day, np.log1p(bars_elapsed), *close_vs_averages], axis=1)

    close_, high_ = close[1:], high[1:]
    n = len(close_)
    labels = np.full(n, np.nan)
    for i in range(n):
        stop = min(i + horizon, n - 1)                       # last row the horizon reaches
        ahead = high_[i + 1:stop + 1][days[i + 1:stop + 1] == days[i]]
        if ahead.size and ahead.max() >= close_[i] * (1.0 + target):
            labels[i] = 1.0
        elif i + horizon > n - 1 and days[i] == days[-1]:
            labels[i] = np.nan                               # this day may continue past the file
        else:
            labels[i] = 0.0                                  # includes bars whose day simply ran out
    # No label where a moving average is still undefined, so `windows` never ends on such a row;
    # a window is one day long, so none of its rows are undefined either.
    labels[np.isnan(features).any(axis=1)] = np.nan

    return features.astype(np.float32), labels.astype(np.float32)


def windows(features, labels, days, minutes, window: int):
    """Every `window`-minute stretch of rows *from a single day*, labelled by its final row.

    Rows are 1-minute bars (see `to_minute_bars`), so a window is `window` rows -- but
    only kept when those rows are `window` consecutive minutes, per `minutes`. A window
    over a gap in the data would cover more time than it claims to, and the model would
    see a jump it couldn't tell from a real one-minute move.

    A window is kept only when its first row is at least `window` bars into the day its
    last row belongs to. That puts the whole window inside one working day and past the
    day's opening bar, so no row in it was derived by comparing today with yesterday.
    The first `window` minutes of each session therefore produce no sample -- there is
    not yet enough of the day to look back over.

    Returns X of shape [samples, features, window] — channels-first, as Conv1d wants —
    and the index of each window's last row, so splits can stay chronological.
    """
    last_valid = np.flatnonzero(~np.isnan(labels))
    elapsed = bars_into_day(days)
    ends = last_valid[elapsed[last_valid] >= window]
    ends = ends[minutes[ends] - minutes[ends - window + 1] == window - 1]
    if ends.size == 0:                                       # no day long enough for one window
        return np.empty((0, features.shape[1], window), dtype=features.dtype), labels[ends], ends
    x = np.stack([features[end - window + 1:end + 1].T for end in ends])
    return x, labels[ends], ends


class Dataset(NamedTuple):
    """One file's windows, and enough of the file to point a window back at its rows."""
    path: str
    x: np.ndarray
    y: np.ndarray
    rows: np.ndarray                                         # the price row each window ends on
    timestamps: list
    prices: np.ndarray


def dataset_from_csv(path: str, window: int, horizon: int, target: float,
                     moving_averages=DEFAULT_MOVING_AVERAGES):
    """One file's dataset, plus a line describing what was in it."""
    timestamps, prices = read_csv(path)
    days, time_of_day = calendar_columns(timestamps)
    unique_days, bars_per_day = np.unique(days, return_counts=True)
    features, labels = features_and_labels(prices, days, time_of_day, horizon, target, moving_averages)
    x, y, ends = windows(features, labels, days[1:], minute_number(days, time_of_day)[1:], window)
    # Features start at row 1 -- row 0 has no previous bar to take a return from -- so a
    # window ending at feature `end` is a window ending at price row `end + 1`.
    dataset = Dataset(path, x, y, ends + 1, timestamps, prices)
    positive = y.mean() if len(y) else float("nan")
    median_bars = int(np.median(bars_per_day)) if len(bars_per_day) else 0
    return dataset, (f"{len(prices)} minutes in session over {len(unique_days)} calendar days, "
                     f"median {median_bars} per day, "
                     f"{len(x)} windows, {positive:.2%} positive")


def chronological_split(x, y, window: int, horizon: int, fractions=(0.7, 0.15)):
    """Train/validation/test in time order, with a gap so no window straddles a boundary.

    Shuffling would leak: overlapping windows share rows, and a label peeks `horizon`
    rows into the future. The embargo of window + horizon samples removes both.
    """
    return [(x[s], y[s]) for s in split_slices(len(x), window, horizon, fractions)]


def split_slices(n: int, window: int, horizon: int, fractions=(0.7, 0.15)):
    """Where the three splits fall, so labels and their provenance can be cut the same way."""
    train_end = int(n * fractions[0])
    val_end = int(n * (fractions[0] + fractions[1]))
    gap = window + horizon
    return [slice(0, train_end),
            slice(train_end + gap, val_end),
            slice(val_end + gap, n)]


def combined_splits(datasets, window: int, horizon: int):
    """Split each file in time order, then concatenate the like-named parts.

    Splitting per file rather than concatenating the files first keeps every file
    represented in every split, and stops a window -- or a label's look-ahead -- from
    reaching across a boundary into an unrelated instrument's bars.

    Alongside each split come its origins: the file and price row every window ended on,
    cut by the same slices, which is what lets a prediction be traced back to the bars
    that produced it.
    """
    parts, origins = [[] for _ in range(3)], [[] for _ in range(3)]
    for file_index, dataset in enumerate(datasets):
        for part, piece in enumerate(split_slices(len(dataset.x), window, horizon)):
            rows = dataset.rows[piece]
            parts[part].append((dataset.x[piece], dataset.y[piece]))
            origins[part].append(np.stack([np.full(len(rows), file_index, dtype=np.int64), rows], axis=1))
    splits = [(np.concatenate([x for x, _ in group]), np.concatenate([y for _, y in group]))
              for group in parts]
    return splits, [np.concatenate(group) for group in origins]


def normalise(splits):
    """Z-score every feature channel using training statistics only."""
    train_x = splits[0][0]
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    std = train_x.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return [((x - mean) / std, y) for x, y in splits], mean, std


class StockCnn(nn.Module):
    """Dilated 1D convolutions: each block doubles the receptive field over the window."""

    def __init__(self, in_channels: int, channels: int = 32, blocks: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        for block in range(blocks):
            dilation = 2 ** block
            layers += [
                nn.Conv1d(in_channels if block == 0 else channels, channels,
                          kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.convolutions = nn.Sequential(*layers)
        # Mean and max pooling over time: "how much of this pattern" and "was it ever there".
        self.head = nn.Linear(channels * 2, 1)

    def forward(self, x):
        h = self.convolutions(x)                       # [batch, channels, window]
        pooled = torch.cat([h.mean(dim=2), h.max(dim=2).values], dim=1)
        return self.head(pooled).squeeze(1)            # [batch] logits


def roc_auc(labels, scores):
    """Probability a random positive scores above a random negative (ties count a half)."""
    positives, negatives = labels.sum(), (1 - labels).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average the ranks of tied scores, else the AUC is biased by sort order.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    summed = np.zeros(len(counts))
    np.add.at(summed, inverse, ranks)
    ranks = (summed / counts)[inverse]
    return (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate(model, x, y, device, thresholds=(0.3, 0.5, 0.7)):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    report = {"auc": roc_auc(y, probabilities), "base_rate": float(y.mean()), "thresholds": {}}
    for threshold in thresholds:
        predicted = probabilities >= threshold
        true_positives = float((predicted & (y == 1)).sum())
        precision = true_positives / max(predicted.sum(), 1)
        recall = true_positives / max((y == 1).sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        report["thresholds"][threshold] = {"precision": precision, "recall": recall, "f1": f1,
                                           "signals": int(predicted.sum())}
    return report, probabilities


def write_hits(path: str, datasets, origins, labels, probabilities, top: int, margin: int, label: int = 1):
    """Write the `top` most confident test windows whose label was `label`, with `margin` rows
    of context either side.

    With the default `label` of 1 a hit is a window the model got right -- a true positive.
    With 0 it is one the model got wrong -- a false positive, the confident call on a move
    that never came. Either way hits are taken in order of the model's probability. What gets written is the bar the window ended on, which
    is the bar the prediction was made from, surrounded by the rows before and after it so the
    move the model spotted can be read off the file.

    Neighbouring bars tend to score alike, so the best hits cluster around the same move. A
    hit whose context would overlap one already chosen from the same file is skipped, which
    keeps every row in the output to a single group and spreads the `top` over distinct moves.

    The context stops at the ends of the hit's own day, so reading a hit's rows off the file
    never puts the previous session's prices next to this one's.
    """
    candidates = np.flatnonzero(labels == label)
    candidates = candidates[np.argsort(-probabilities[candidates], kind="stable")]
    chosen = []
    for index in candidates:
        if len(chosen) == top:
            break
        file_index, row = origins[index]
        if all(other_file != file_index or abs(other_row - row) > 2 * margin
               for other_file, other_row in (origins[c] for c in chosen)):
            chosen.append(index)

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hit", "source", "offset", "probability",
                         "timestamp", "open", "high", "low", "close", "volume"])
        for hit, index in enumerate(chosen):
            file_index, row = origins[index]
            dataset = datasets[file_index]
            day = as_utc(dataset.timestamps[row]).date()
            for context in range(max(row - margin, 0), min(row + margin + 1, len(dataset.prices))):
                if as_utc(dataset.timestamps[context]).date() != day:
                    continue
                writer.writerow([hit, dataset.path, context - row, f"{probabilities[index]:.6f}",
                                 dataset.timestamps[context].isoformat(), *dataset.prices[context]])
    return len(chosen)


def train(model, train_split, validation_split, device, epochs: int, batch_size: int, learning_rate: float):
    """Plain BCE training, weighted towards the rare positive class, keeping the best validation AUC."""
    x_train, y_train = (torch.from_numpy(a) for a in train_split)
    positives = float(y_train.sum())
    pos_weight = torch.tensor([(len(y_train) - positives) / max(positives, 1.0)], device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_auc, best_state = -math.inf, None
    for epoch in range(1, epochs + 1):
        model.train()
        # Batches are shuffled; the split itself stayed chronological, so nothing leaks.
        order = torch.randperm(len(x_train))
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            inputs, targets = x_train[batch].to(device), y_train[batch].to(device)
            optimiser.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)

        report, _ = evaluate(model, *validation_split, device)
        print(f"epoch {epoch:3d}  loss {total / len(order):.4f}  validation AUC {report['auc']:.4f}")
        if report["auc"] > best_auc:
            best_auc = report["auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"best validation AUC {best_auc:.4f}")
    return model

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="comma-separated timestamp,open,high,low,close,volume files, "
                                      "all trained and evaluated as one model "
                                      "(default: generate synthetic data)")
    parser.add_argument("--window", type=int, default=64,
                        help="minutes of history the CNN sees; bars are merged into 1-minute bars as "
                             "they are read, and windows over a missing minute are skipped")
    parser.add_argument("--horizon", type=int, default=20,
                        help="rows ahead the 2%% move must happen in, capped at the end of the calendar day")
    parser.add_argument("--target", type=float, default=0.02, help="the move to predict, as a fraction")
    parser.add_argument("--moving-averages", default=",".join(map(str, DEFAULT_MOVING_AVERAGES)),
                        help="comma-separated day counts; each adds close relative to the moving average "
                             "of that many previous days' closes as a feature (empty for none). Days "
                             "before the longest average is available produce no samples")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--thresholds", default="0.3,0.5,0.7",
                        help="comma-separated probabilities at which to report precision/recall on the test set")
    parser.add_argument("--hits", help="write the correctly predicted test rows here, as CSV, and the most confident wrong ones alongside with _negative appended to the name")
    parser.add_argument("--top", type=int, default=10,
                        help="how many of the most confident, non-overlapping predictions --hits writes to each file")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", help="where to write the trained weights and normalisation statistics")
    args = parser.parse_args()

    try:
        thresholds = sorted(float(t) for t in args.thresholds.split(",") if t.strip())
    except ValueError:
        parser.error(f"--thresholds wants numbers, got {args.thresholds!r}")
    if not thresholds:
        parser.error("--thresholds needs at least one value")
    try:
        moving_averages = [int(n) for n in args.moving_averages.split(",") if n.strip()]
    except ValueError:
        parser.error(f"--moving-averages wants whole numbers, got {args.moving_averages!r}")
    if any(n < 1 for n in moving_averages):
        parser.error("--moving-averages must all be at least 1")

    torch.manual_seed(args.seed)
    paths = [path.strip() for path in args.csv.split(",") if path.strip()]

    datasets = []
    for path in paths:
        dataset, summary = dataset_from_csv(path, args.window, args.horizon, args.target, moving_averages)
        print(f"{path}: {summary}")
        if len(dataset.x) == 0:
            print(f"  skipped: no {args.window} unbroken minutes of labelled rows "
                  f"between {SESSION_START:%H:%M} and {SESSION_END:%H:%M} UTC"
                  + (f" after the first {max(moving_averages)} days" if moving_averages else ""))
            continue
        datasets.append(dataset)
    if not datasets:
        raise SystemExit(f"none of the files held a full window of labelled rows between "
                         f"{SESSION_START:%H:%M} and {SESSION_END:%H:%M} UTC")

    positives = sum(float(dataset.y.sum()) for dataset in datasets)
    total = sum(len(dataset.y) for dataset in datasets)
    print(f"{total} windows of shape {datasets[0].x.shape[1:]} from {len(datasets)} "
          f"file{'s' if len(datasets) > 1 else ''}, {positives / total:.2%} of them positive")
    if positives == 0:
        raise SystemExit("no positive examples: with one bar per day (or coarser) the same-day "
                         "constraint can never be met, so the label is always 0")

    splits, origins = combined_splits(datasets, args.window, args.horizon)
    splits, mean, std = normalise(splits)
    for name, (split_x, split_y) in zip(("train", "validation", "test"), splits):
        print(f"{name:10s} {len(split_x):6d} windows, {split_y.mean():.2%} positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StockCnn(in_channels=splits[0][0].shape[1]).to(device)
    train(model, splits[0], splits[1], device, args.epochs, args.batch_size, args.learning_rate)

    report, probabilities = evaluate(model, *splits[2], device, thresholds)
    print(f"\ntest AUC {report['auc']:.4f} against a base rate of {report['base_rate']:.2%}")
    for threshold, scores in report["thresholds"].items():
        print(f"  p>={threshold}: {scores['signals']:5d} signals, "
              f"precision {scores['precision']:.2%}, recall {scores['recall']:.2%}, F1 {scores['f1']:.3f}")

    if args.hits:
        # The model's most confident calls are the ones worth looking at row by row.
        written = write_hits(args.hits, datasets, origins[2], splits[2][1], probabilities,
                             args.top, args.horizon)
        print(f"wrote the top {written} non-overlapping correct predictions "
              f"(with {args.horizon} rows either side) to {args.hits}")
        # And the calls it was just as sure of but got wrong, for comparison.
        stem, extension = os.path.splitext(args.hits)
        negative_path = f"{stem}_negative{extension}"
        written = write_hits(negative_path, datasets, origins[2], splits[2][1], probabilities,
                             args.top, args.horizon, label=0)
        print(f"wrote the top {written} non-overlapping wrong predictions "
              f"(with {args.horizon} rows either side) to {negative_path}")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                    "window": args.window, "horizon": args.horizon, "target": args.target,
                    "features": feature_names(moving_averages), "moving_averages": moving_averages,
                    "sources": paths}, args.save)
        print(f"saved to {args.save}")


if __name__ == "__main__":
    main()
