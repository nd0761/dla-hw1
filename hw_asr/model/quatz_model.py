from torch import nn
import torch
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from hw_asr.base import BaseModel


def init_activation(dropout_proba=0.2):
    return [nn.ReLU(), nn.Dropout(dropout_proba)]


def init_basic_module(
        in_channels, out_channels, kernel_size,
        batch_eps=1e-3, dropout_proba=0.2
):
    layers = [
        nn.Conv1d(in_channels, out_channels, kernel_size),
        nn.BatchNorm1d(out_channels, eps=batch_eps),
    ]
    layers.extend(init_activation(dropout_proba))
    return layers


def init_first_module_tsc(
        in_channels, k, out_channels, kernel_size
):
    return [
        nn.Conv1d(in_channels, in_channels * k,
                  kernel_size, groups=in_channels),           # depthwise
        nn.Conv1d(in_channels, out_channels, kernel_size=1),  # pointwise
        nn.BatchNorm1d(out_channels, eps=1e-3)
    ]


def init_residual_module(in_channels, out_channels, kernel_size):
    modules = [
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size),
        nn.BatchNorm1d(out_channels, eps=1e-3)
    ]
    return modules


def init_tsc_conv(
        repeat,
        in_channels, out_channels,
        dropout_proba=0.2, kernel_size=1
):
    rinse_and_repeat_modules = []
    current_channel = in_channels
    for _ in range(repeat):
        rinse_and_repeat_modules.extend(
            init_first_module_tsc(
                current_channel, 1,
                out_channels, kernel_size=kernel_size)
        )
        rinse_and_repeat_modules.extend(init_activation(dropout_proba))
        current_channel = out_channels
    return rinse_and_repeat_modules


class TcsBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels,
            repeat=3, dropout_proba=0.2
    ):
        super(TcsBlock, self).__init__()
        current_channels = in_channels
        modules = init_tsc_conv(
            repeat, in_channels, out_channels,
            dropout_proba, kernel_size=1
        )
        self.model = nn.Sequential(*modules)
        residual_modules = init_residual_module(
            in_channels, out_channels, kernel_size=1
        )
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),  # pointwise
            nn.BatchNorm1d(out_channels, eps=1e-3)
        )
        self.out = nn.Sequential(*init_activation())

    def forward(self, x):
        return self.out(self.model(x) + self.residual(x))


class QuartzNetModel(BaseModel):
    def __init__(self, n_feats, n_class, repeat, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.first_block = nn.Sequential(*init_basic_module(
            n_feats, n_feats, 1, dropout_proba=0.2
        ))
        self.tcss = []
        current_channels = n_feats
        for _ in range(repeat):
            new_channels = current_channels * 2
            self.tcss.append(TcsBlock(
                current_channels, new_channels, repeat=2
            ))
            current_channels = new_channels
        self.final_blocks = [
            nn.Sequential(*init_basic_module(
                current_channels, current_channels, 1, dropout_proba=0.2
            )),
            nn.Sequential(*init_basic_module(
                current_channels, current_channels, 1, dropout_proba=0.2
            ))
        ]
        self.point_wise = nn.Sequential(
            nn.Conv1d(current_channels, n_class, kernel_size=1),  # pointwise
            nn.BatchNorm1d(n_class, eps=1e-3),
            nn.Dropout(p=0.2)
        )

    def forward(self, spectrogram, *args, **kwargs):
        packed_inputs = pack_padded_sequence(spectrogram, kwargs["spectrogram_length"],
                                             enforce_sorted=False, batch_first=True)

        out = self.first_block(packed_inputs)

        final_out = out
        for tcs in self.tcss:
            out = tcs(final_out)
            final_out += out

        for block in self.final_blocks:
            final_out = block(final_out)

        out, _ = pad_packed_sequence(final_out, batch_first=True)
        out = self.point_wise(out)
        return {"logits": out}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
