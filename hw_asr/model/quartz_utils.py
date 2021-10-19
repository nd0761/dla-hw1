from torch import nn


def get_padding(kernel, dilation):
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
