"""A 1D CNN that flags bars from which price is likely to rise 2% within the next H minutes.

Run against real data:
    python StockCnn.py --csv prices.csv
Several files train (and are evaluated as) one model -- the features are scale-free, so
bars from different instruments are comparable:
    python StockCnn.py --csv aapl.csv,msft.csv,nvda.csv
Or a real example:
    python StockCnn.py --csv /home/henryp/Downloads/aapl_dataset_London-Strategic-Edge.csv --target 0.02 --horizon 60

The data, labels and training are in StockCommon.py.
"""
import torch
from torch import nn

import StockCommon


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


if __name__ == "__main__":
    StockCommon.main(StockCnn, __doc__)
