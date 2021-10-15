from torch import nn
import torch
from torch.nn import Sequential

from hw_asr.base import BaseModel


class RnnModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.rnn = nn.RNN(n_feats, fc_hidden, 2)
        self.fc = nn.Linear(fc_hidden, n_class)

    def forward(self, spectrogram, *args, **kwargs):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
        out, hidden = self.rnn(spectrogram, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return {"logits": self.net(spectrogram)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
