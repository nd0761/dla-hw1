from torch import nn
import torch
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from hw_asr.base import BaseModel


class RnnModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, fc_hidden, *args, **kwargs)
        self.n_layers = kwargs['n_layers']
        self.fc_hidden = fc_hidden
        self.rnn = nn.RNN(n_feats, fc_hidden, self.n_layers, batch_first=True)
        self.fc = nn.Linear(in_features=fc_hidden, out_features=n_class)

    def forward(self, spectrogram, *args, **kwargs):
        packed_inputs = pack_padded_sequence(spectrogram, kwargs["spectrogram_length"],
                                             enforce_sorted=False, batch_first=True)

        out, _ = self.rnn(packed_inputs)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out)

        temp = out.shape
        return {"logits": out}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
