import torch
import torch.nn as nn
import torch.nn.functional as F

from hw_asr.base import BaseModel
import hw_asr.model.utils as qu


class MaskedConv1d(nn.Conv1d):
    __constants__ = ["masked"]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, masked=True):
        super(MaskedConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.masked = masked

    def get_seq_len(self, lens):
        return ((lens + 2 * self.padding[0] - self.dilation[0]
                 * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1)

    def forward(self, x, x_lens=None):
        if self.masked:
            max_len = x.size(2)
            idxs = torch.arange(max_len, dtype=x_lens.dtype, device=x_lens.device)
            mask = idxs.expand(x_lens.size(0), max_len) >= x_lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
            x_lens = self.get_seq_len(x_lens)

        return super(MaskedConv1d, self).forward(x), x_lens