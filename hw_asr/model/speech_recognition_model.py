from torch import nn
import torch
from torch.nn import Sequential

from hw_asr.base import BaseModel
import hw_asr.model.quartz_utils as qu
import torch.nn.functional as F
# https://arxiv.org/pdf/1603.05027.pdf


def init_residual_modules(in_channels, out_channels, kernel, stride, dropout, n_feats):
    return [
        nn.LayerNorm(n_feats),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding=qu.get_padding(kernel))
    ]


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(Residual, self).__init__()
        
        modules = init_residual_modules(in_channels, out_channels, kernel, stride, dropout, n_feats)
        modules.extend(init_residual_modules(out_channels, out_channels, kernel, stride, dropout, n_feats))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        out = self.net(x)
        return out + x


class Bidirectional(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(Bidirectional, self).__init__()

        modules = [
            nn.LayerNorm(rnn_dim),
            nn.GELU()
        ]
        self.net = nn.Sequential(*modules)
        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.net(x)
        out, _ = self.BiGRU(out)
        return self.dropout(out)


class SpeechRecognitionModel(BaseModel):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.2, *args, **kwargs):

        super().__init__(n_feats, n_class, *args, **kwargs)
        n_feats = n_feats // 2
        kernel = 3
        current_channels = 32
        residual_modules = [
            nn.Conv2d(1, current_channels, kernel, stride=stride, padding=qu.get_padding(kernel))
        ]

        residual_modules.extend([
            Residual(current_channels, current_channels,
                     kernel, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.residual = nn.Sequential(*residual_modules)

        current_channels = n_feats * 32
        bidir_modules = [nn.Linear(current_channels, rnn_dim)]

        bidir_modules.extend([
            Bidirectional(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                          hidden_size=rnn_dim, dropout=dropout, batch_first=(i == 0))
            for i in range(n_rnn_layers)
        ])
        self.bidir = nn.Sequential(*bidir_modules)
        current_channels = rnn_dim * 2

        self.fc = nn.Sequential(
            nn.Linear(current_channels, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        spectrogram = spectrogram.unsqueeze(1)
        temp=spectrogram.shape
        out = self.residual(spectrogram)

        sizes = out.size()
        temp1=out.shape
        out = out.transpose(1, 2).contiguous()
        out = out.view(sizes[0], sizes[2],  sizes[1] * sizes[3])
        temp2=out.shape

        out = self.bidir(out)
        return self.fc(out)

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
