"""An LSTM that flags bars from which price is likely to rise 2% within the next H minutes.

Run against real data:
    python StockLstm.py --csv prices.csv
Several files train (and are evaluated as) one model -- the features are scale-free, so
bars from different instruments are comparable:
    python StockLstm.py --csv aapl.csv,msft.csv,nvda.csv
Or a real example:
    python StockLstm.py --csv /home/henryp/Downloads/aapl_dataset_London-Strategic-Edge.csv --target 0.02 --horizon 60

The data, labels and training are in StockCommon.py.
"""
import torch
from torch import nn

import StockCommon


class StockLstm(nn.Module):
    """A stacked LSTM read over the window oldest bar first, classifying from where it ends up.

    The prediction is made from the window's last bar, so the last time step's hidden
    state -- the whole window folded into what matters as of now -- is what the head sees.
    """

    def __init__(self, in_channels: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # Windows arrive channels-first, [batch, features, window]; the LSTM wants time second.
        outputs, _ = self.lstm(x.transpose(1, 2))      # [batch, window, hidden]
        return self.head(self.dropout(outputs[:, -1])).squeeze(1)   # [batch] logits


if __name__ == "__main__":
    StockCommon.main(StockLstm, __doc__)
