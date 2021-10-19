from torch import nn
import torch

import hw_asr.model.quartz_utils as quartz_utils

from hw_asr.base import BaseModel


class TcsBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels,
            repeat=3, kernel_size=1,
            padding=0
    ):
        super(TcsBlock, self).__init__()

        modules = quartz_utils.init_tsc_conv(
            repeat, in_channels, out_channels,
            kernel_size=kernel_size, padding=padding
        )
        self.model = nn.Sequential(*modules)

        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(1,), bias=False),  # pointwise
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
            repeat_b_block,
            *args, **kwargs
    ):
        super().__init__(n_feats, n_class, *args, **kwargs)
        c1_modules = quartz_utils.init_first_module_tsc(
            n_feats, 1, output_channels[0], kernels[0], stride=2,
            padding=quartz_utils.get_padding(kernels[0], dilation=1)
        )
        c1_modules.append(nn.ReLU())
        self.first_block = nn.Sequential(*c1_modules)

        tcss_modules = []
        current_channels = output_channels[0]
        self.repeat = repeat
        for i in range(repeat):
            for _ in range(repeat_b_block[i]):
                new_channels = output_channels[i + 1]
                tcss_modules.append(TcsBlock(
                    current_channels, new_channels,
                    repeat=tcs_repeat, kernel_size=kernels[i + 1],
                    padding=quartz_utils.get_padding(kernels[i + 1], dilation=1)
                ))
                current_channels = new_channels

        self.tcss = (nn.Sequential(*tcss_modules))

        c2_modules = quartz_utils.init_first_module_tsc(
            current_channels, 1, output_channels[-2], kernels[-2],
            dilation=2,
            padding=quartz_utils.get_padding(kernels[-2], dilation=2)
        )
        c2_modules.append(nn.ReLU())

        c3_modules = quartz_utils.init_first_module_tsc(
            output_channels[-2], 1, output_channels[-1], kernels[-1],
            padding=0
        )
        c3_modules.append(nn.ReLU())

        final_blocks_modules = [
            nn.Sequential(*c2_modules),
            nn.Sequential(*c3_modules)
        ]
        self.final_blocks = nn.Sequential(*final_blocks_modules)

        self.fc = nn.Sequential(
            nn.Conv1d(output_channels[-1], n_class, kernel_size=(1,), dilation=(2,))
        )

    def forward(self, spectrogram, *args, **kwargs):
        spectrogram = torch.transpose(spectrogram, 1, 2)

        out = self.first_block(spectrogram)

        for tcs in self.tcss:
            out = tcs(out)
        for block in self.final_blocks:
            out = block(out)

        out = self.fc(out)
        out = torch.transpose(out, 1, 2)
        return {"logits": out}

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2  # we reduce time dimension here
