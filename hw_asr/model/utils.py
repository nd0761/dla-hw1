from torch import nn
from hw_asr.model.masked_conv import MaskedConv1d


activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


def init_conv_bn(in_channels, out_channels, **kwargs):
    return [MaskedConv1d(in_channels, out_channels, masked=True, **kwargs),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)]


def init_act_dropout(activation, dropout=0.2):
    return [activation, nn.Dropout(p=dropout)]


def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == MaskedConv1d:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
    elif type(m) == nn.BatchNorm1d:
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def get_padding(kernel, dilation=1):
    if dilation > 1:
        return (dilation * kernel) // 2 - 1
    return kernel // 2


def init_first_module_tsc(
        in_channels, k, out_channels, kernel_size,
        stride=(1,), batch_eps=1e-3, padding=(0,), dilation=(1,)
):
    return [
        nn.Conv1d(in_channels, in_channels * k,
                  kernel_size, groups=in_channels, stride=stride,
                  bias=False, padding=padding, dilation=dilation),  # depthwise
        nn.Conv1d(in_channels, out_channels, kernel_size=(1,), bias=False),  # pointwise
        nn.BatchNorm1d(out_channels, eps=batch_eps)
    ]


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
