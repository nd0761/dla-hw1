from torch import nn
import torch
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from hw_asr.base import BaseModel


def init_first_module_tsc(
        in_channels, k, out_channels, kernel_size,
        stride=1, batch_eps=1e-3, padding=0
):
    return [
        nn.Conv1d(in_channels, in_channels * k,
                  kernel_size, groups=in_channels, stride=stride,
                  bias=False, padding=padding),  # depthwise
        nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),  # pointwise
        nn.BatchNorm1d(out_channels, eps=batch_eps)
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
        kernel_size=1, padding=0
):
    rinse_and_repeat_modules = []
    current_channel = in_channels
    for i in range(repeat):
        rinse_and_repeat_modules.extend(
            init_first_module_tsc(
                current_channel, 1,
                out_channels, kernel_size=kernel_size, padding=padding)
        )
        if i != repeat - 1:
            rinse_and_repeat_modules.append(nn.ReLU())
        current_channel = out_channels
    return rinse_and_repeat_modules


class TcsBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels,
            repeat=3, kernel_size=1,
            padding=0
    ):
        super(TcsBlock, self).__init__()

        modules = init_tsc_conv(
            repeat, in_channels, out_channels,
            kernel_size=kernel_size, padding=padding
        )
        self.model = nn.Sequential(*modules)

        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm1d(out_channels, eps=1e-3)
        )
        self.out = nn.Sequential(nn.ReLU())

    def forward(self, x):
        out = self.model(x)
        out += self.residual(x)
        return self.out(out)


class QuartzNetModel(BaseModel):
    def __init__(
            self, n_feats, n_class,
            repeat, tcs_repeat,
            output_channels, kernels,
            paddings,
            *args, **kwargs
    ):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.first_block = nn.Sequential(*init_first_module_tsc(
            n_feats, 1, output_channels[0], kernels[0], stride=2, padding=paddings[0]
        ))
        tcss_modules = []
        current_channels = output_channels[0]
        self.repeat = repeat
        for i in range(repeat):
            new_channels = output_channels[i + 1]
            tcss_modules.append(TcsBlock(
                current_channels, new_channels,
                repeat=tcs_repeat, kernel_size=kernels[i + 1],
                padding=paddings[i + 1]
            ))
            current_channels = new_channels

        self.tcss = (nn.Sequential(*tcss_modules))

        final_blocks_modules = [
            nn.Sequential(*init_first_module_tsc(
                current_channels, 1, output_channels[-2], kernels[-2], padding=paddings[-2]
            )),
            nn.Sequential(*init_first_module_tsc(
                output_channels[-2], 1, output_channels[-1], kernels[-1], padding=paddings[-1]
            ))
        ]
        self.final_blocks = nn.Sequential(*final_blocks_modules)

        self.fc = nn.Sequential(
            nn.Conv1d(output_channels[-1], n_class, kernel_size=1, dilation=2)
        )

    def forward(self, spectrogram, *args, **kwargs):
        packed_inputs = pack_padded_sequence(spectrogram, kwargs["spectrogram_length"],
                                             enforce_sorted=False, batch_first=True)
        spectrogram = torch.transpose(spectrogram, 1, 2)

        out = self.first_block(spectrogram)

        final_out = out
        i = 0
        for tcs in self.tcss:
            out = tcs(final_out)
            if i == 0 or i == self.repeat - 1:
                final_out += out
            else:
                final_out = out
            i += 1

        for block in self.final_blocks:
            final_out = block(final_out)

        # out, _ = pad_packed_sequence(final_out, batch_first=True)
        out = self.fc(final_out)
        return {"logits": out}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
