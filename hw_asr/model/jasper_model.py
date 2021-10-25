import torch
import torch.nn as nn
import torch.nn.functional as F

from hw_asr.base import BaseModel
import hw_asr.model.utils as qu
from hw_asr.model.masked_conv import MaskedConv1d
# Jasper Block https://arxiv.org/pdf/1904.03288.pdf


class JasperBlock(nn.Module):

    def __init__(self, in_channels, filter, repeat=3, kernel=11, stride=1,
                 dilation=1, dropout=0.2, activation=None,
                 residual=True, residual_panes=[]):
        super(JasperBlock, self).__init__()

        assert activation is not None, "Set activation for encoder"

        padding_val = qu.get_padding(kernel, dilation)
        self.conv = nn.ModuleList()
        current_channels = in_channels
        for i in range(repeat):
            self.conv.extend(qu.init_conv_bn(
                current_channels,
                filter,
                kernel_size=kernel,
                stride=stride,
                dilation=dilation,
                padding=padding_val
            ))
            current_channels = filter
            if i != repeat - 1:
                self.conv.extend(qu.init_act_dropout(activation, dropout))

        self.res = None
        if residual:
            self.res = nn.ModuleList()

            res_panes = residual_panes.copy()
            self.dense_residual = residual
            if len(residual_panes) == 0:
                res_panes = [in_channels]
                self.dense_residual = False

            for ip in res_panes:
                self.res.extend(qu.init_conv_bn(ip, filter, kernel_size=1))
        out_modules = qu.init_act_dropout(activation, dropout)
        self.out = nn.Sequential(*out_modules)

    def forward(self, x, lens=None):

        out = x[-1]
        lens_cur = lens
        for i, l in enumerate(self.conv):
            out, lens_cur = l(out, lens_cur)

        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = x[i]
                for j, res_layer in enumerate(layer):
                    if j == 0:
                        res_out, _ = res_layer(res_out, lens)
                    else:
                        res_out = res_layer(res_out)
                out += res_out

        out = self.out(out)
        if self.res is not None and self.dense_residual:
            out = x + [out]
        else:
            out = [out]

        return out, lens


class JasperEncoder(nn.Module):
    def __init__(
            self, n_feats, activation,
            filters, repeat_block, kernels, strides,
            dilations, dropouts, residuals, denses
    ):
        super(JasperEncoder, self).__init__()

        self.layers = nn.ModuleList()
        encoder_activation = qu.activations[activation]()

        all_residual_panes = []
        for i, (filter, repeat, kernel, stride, dilation, dropout, residual, dense) \
                in enumerate(zip(filters, repeat_block, kernels, strides, dilations, dropouts, residuals, denses)):
            residual_panes = []
            if dense:
                all_residual_panes += [n_feats]
                residual_panes = all_residual_panes


            self.layers.append(
                JasperBlock(n_feats, filter, repeat, kernel, stride, dilation,
                            dropout, encoder_activation, residual, residual_panes))

            n_feats = filter

        self.apply(qu.init_weights)

    def forward(self, x, x_lens=None):
        out, out_lens = [x], x_lens
        for l in self.layers:
            out, out_lens = l(out, out_lens)

        return out, out_lens


class JasperDecoderForCTC(nn.Module):
    def __init__(self, n_feats, n_classes):
        super(JasperDecoderForCTC, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(n_feats, n_classes, kernel_size=1, bias=True),
        )
        self.apply(qu.init_weights)

    def forward(self, enc_out):
        out = self.layers(enc_out[-1]).transpose(1, 2)
        return F.log_softmax(out, dim=2)


class Jasper(BaseModel):
    def __init__(self, encoder, decoder, n_feats, n_class, transpose_in=False, *args, **kwargs):
        super(Jasper, self).__init__(n_feats, n_class)
        self.transpose_in = transpose_in
        self.encoder = JasperEncoder(**encoder)
        self.decoder = JasperDecoderForCTC(**decoder, n_classes=n_class)
        self.n_class = n_class

    def forward(self, spectrogram, spectrogram_length, *args, **kwargs):
        spectrogram = spectrogram.transpose(1, 2)

        enc, enc_lens = self.encoder(spectrogram, spectrogram_length)
        out = self.decoder(enc)
        return out

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
