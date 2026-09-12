"""A 1D CNN that flags bars from which price is likely to rise 2% within the next 20 rows.

Reads a CSV of `timestamp,open,high,low,close,volume`, turns it into overlapping
windows of scale-free features, and trains a convolutional classifier on the binary
label "did the high reach close * 1.02 within the next H rows *and* before midnight?".

The same-calendar-day constraint means the effective horizon shrinks as the session
runs down, so the timestamp itself becomes predictive and is fed in as a feature.

Run without arguments for a synthetic random-walk demo:
    python StockCnn.py
Or against real data:
    python StockCnn.py --csv prices.csv
"""
import argparse
import csv
import datetime as dt
import math

import numpy as np
import torch
from torch import nn

FEATURE_NAMES = ["log_return", "high_vs_close", "low_vs_close", "open_vs_close", "log_volume_change",
                 "time_of_day", "bars_elapsed_today"]


def parse_timestamp(raw: str) -> dt.datetime:
    """ISO-8601 (with or without a `T`, `Z` or offset) or an epoch number in seconds/millis."""
    text = raw.strip()
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    number = float(text)
    return dt.datetime.fromtimestamp(number / 1000.0 if number > 1e11 else number, tz=dt.timezone.utc)


def read_csv(path: str):
    """Rows sorted by timestamp: the parsed timestamps and the OHLCV columns as floats.

    The timestamps are no longer just for ordering — the calendar day decides where
    each label's look-ahead window is cut off.
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    timestamps = sorted(parse_timestamp(row["timestamp"]) for row in rows)
    rows.sort(key=lambda row: parse_timestamp(row["timestamp"]))
    prices = np.array([[float(row[c]) for c in ("open", "high", "low", "close", "volume")] for row in rows],
                      dtype=np.float64)
    return timestamps, prices


def calendar_columns(timestamps):
    """Day number (for grouping) and time of day as a fraction, per row.

    Mixed time zones in one file would put bars in the wrong day, so if the stamps
    are offset-aware they are all converted to the first one's zone before the date
    is taken.
    """
    if timestamps and timestamps[0].tzinfo is not None:
        zone = timestamps[0].tzinfo
        timestamps = [t.astimezone(zone) for t in timestamps]
    days = np.array([t.date().toordinal() for t in timestamps], dtype=np.int64)
    seconds = np.array([t.hour * 3600 + t.minute * 60 + t.second for t in timestamps], dtype=np.float64)
    return days, seconds / 86400.0


def features_and_labels(prices, days, time_of_day, horizon: int, target: float):
    """Per-row features plus the forward-looking label, aligned on the same index.

    Price features are ratios/differences so the net never sees the absolute price
    level: a model trained on a $10 stock should still work when it trades at $200.
    Row 0 is dropped because the return features need a previous row.

    The label for row i is 1 when some bar in rows i+1 .. i+horizon *that falls on the
    same calendar day as row i* trades at or above close[i] * (1 + target). Rows near
    the close therefore have fewer chances, which is why the clock is a feature: with
    no notion of time the model could only average over "how much day is left".
    """
    open_, high, low, close, volume = (prices[:, i] for i in range(5))

    log_return = np.diff(np.log(close))
    high_vs_close = (high[1:] / close[1:]) - 1.0
    low_vs_close = (low[1:] / close[1:]) - 1.0
    open_vs_close = (open_[1:] / close[1:]) - 1.0
    log_volume_change = np.diff(np.log1p(volume))

    days, time_of_day = days[1:], time_of_day[1:]
    # How far into its day each bar is, counted in bars: the clock as the data actually
    # samples it, which is what limits how many chances the 2% move has left.
    _, first_of_day = np.unique(days, return_index=True)
    bars_elapsed = np.arange(len(days)) - np.repeat(first_of_day, np.diff(np.append(first_of_day, len(days))))

    features = np.stack([log_return, high_vs_close, low_vs_close, open_vs_close, log_volume_change,
                         time_of_day, np.log1p(bars_elapsed)], axis=1)

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

    return features.astype(np.float32), labels.astype(np.float32)


def windows(features, labels, window: int):
    """Every window of `window` consecutive rows, labelled by its final row.

    Returns X of shape [samples, features, window] — channels-first, as Conv1d wants —
    and the index of each window's last row, so splits can stay chronological.
    """
    last_valid = np.flatnonzero(~np.isnan(labels))
    ends = last_valid[last_valid >= window - 1]
    x = np.stack([features[end - window + 1:end + 1].T for end in ends])
    return x, labels[ends], ends


def chronological_split(x, y, window: int, horizon: int, fractions=(0.7, 0.15)):
    """Train/validation/test in time order, with a gap so no window straddles a boundary.

    Shuffling would leak: overlapping windows share rows, and a label peeks `horizon`
    rows into the future. The embargo of window + horizon samples removes both.
    """
    n = len(x)
    train_end = int(n * fractions[0])
    val_end = int(n * (fractions[0] + fractions[1]))
    gap = window + horizon
    slices = [slice(0, train_end),
              slice(train_end + gap, val_end),
              slice(val_end + gap, n)]
    return [(x[s], y[s]) for s in slices]


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


def synthetic_csv(path: str, days: int = 52, bars_per_day: int = 390, seed: int = 0):
    """A random walk in which a volume spike precedes a burst of upward drift.

    Purely to have something learnable to demo against: the signal is a spike
    followed `lead` bars later by positive drift, which is exactly the sort of
    local, shift-invariant pattern a CNN over a window should be able to find.
    Bars are one minute apart from 09:30, so the calendar day actually bites.
    """
    rng = np.random.default_rng(seed)
    session_start = dt.datetime(2024, 1, 1, 9, 30)
    price, lead = 100.0, 5
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for day in range(days):
            drift, countdown = 0.0, 0                        # each session starts flat
            start = session_start + dt.timedelta(days=day)
            for bar in range(bars_per_day):
                spike = countdown == 0 and rng.random() < 0.004
                if spike:
                    countdown = lead
                elif countdown > 0:
                    countdown -= 1
                    if countdown == 0:
                        drift = 0.0025
                drift *= 0.93
                open_ = price
                price = max(price * (1 + rng.normal(drift, 0.003)), 1e-3)
                high = max(open_, price) * (1 + abs(rng.normal(0, 0.002)))
                low = min(open_, price) * (1 - abs(rng.normal(0, 0.002)))
                volume = rng.lognormal(10, 0.4) * (6.0 if spike else 1.0)
                writer.writerow([(start + dt.timedelta(minutes=bar)).isoformat(),
                                 f"{open_:.4f}", f"{high:.4f}", f"{low:.4f}", f"{price:.4f}", f"{volume:.1f}"])
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="timestamp,open,high,low,close,volume (default: generate synthetic data)")
    parser.add_argument("--window", type=int, default=64, help="rows of history the CNN sees")
    parser.add_argument("--horizon", type=int, default=20,
                        help="rows ahead the 2%% move must happen in, capped at the end of the calendar day")
    parser.add_argument("--target", type=float, default=0.02, help="the move to predict, as a fraction")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", help="where to write the trained weights and normalisation statistics")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    path = args.csv or synthetic_csv("/tmp/synthetic_prices.csv")
    if not args.csv:
        print(f"no --csv given, generated synthetic data at {path}")

    timestamps, prices = read_csv(path)
    days, time_of_day = calendar_columns(timestamps)
    unique_days, bars_per_day = np.unique(days, return_counts=True)
    print(f"{len(prices)} rows over {len(unique_days)} calendar days, "
          f"median {int(np.median(bars_per_day))} bars per day")

    features, labels = features_and_labels(prices, days, time_of_day, args.horizon, args.target)
    x, y, _ = windows(features, labels, args.window)
    print(f"{len(x)} windows of shape {x.shape[1:]}, {y.mean():.2%} of them positive")
    if y.sum() == 0:
        raise SystemExit("no positive examples: with one bar per day (or coarser) the same-day "
                         "constraint can never be met, so the label is always 0")

    splits, mean, std = normalise(chronological_split(x, y, args.window, args.horizon))
    for name, (split_x, split_y) in zip(("train", "validation", "test"), splits):
        print(f"{name:10s} {len(split_x):6d} windows, {split_y.mean():.2%} positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StockCnn(in_channels=x.shape[1]).to(device)
    train(model, splits[0], splits[1], device, args.epochs, args.batch_size, args.learning_rate)

    report, _ = evaluate(model, *splits[2], device)
    print(f"\ntest AUC {report['auc']:.4f} against a base rate of {report['base_rate']:.2%}")
    for threshold, scores in report["thresholds"].items():
        print(f"  p>={threshold}: {scores['signals']:5d} signals, "
              f"precision {scores['precision']:.2%}, recall {scores['recall']:.2%}, F1 {scores['f1']:.3f}")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                    "window": args.window, "horizon": args.horizon, "target": args.target,
                    "features": FEATURE_NAMES}, args.save)
        print(f"saved to {args.save}")


if __name__ == "__main__":
    main()
