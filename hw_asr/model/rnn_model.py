from torch import nn
import torch
from torch.nn import Sequential

from hw_asr.base import BaseModel


class RnnModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, fc_hidden, *args, **kwargs)
        self.n_layers = kwargs['n_layers']
        self.fc_hidden = fc_hidden
        self.rnn = nn.RNN(n_feats, fc_hidden, self.n_layers, batch_first=True)
        self.fc = nn.Linear(in_features=fc_hidden, out_features=n_class)

    def forward(self, spectrogram, *args, **kwargs):
        out, _ = self.rnn(spectrogram)

        out = self.fc(out)
        return {"logits": out}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
