from torch import nn
import torch
from torch.nn import Sequential

from hw_asr.base import BaseModel


class LstmModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, fc_hidden, *args, **kwargs)
        self.n_layers = kwargs['n_layers']
        self.fc_hidden = fc_hidden
        self.lstm = nn.LSTM(n_feats, fc_hidden, self.n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=2 * fc_hidden, out_features=n_class)

    def forward(self, spectrogram, *args, **kwargs):
        out, _ = self.lstm(spectrogram)

        out = self.fc(out)
        return {"logits": out}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
