import torch

from torch.nn import Sequential
import torchaudio


from hw_asr.augmentations.base import AugmentationBase


class AddSpecAugment(AugmentationBase):
    def __init__(self, freq_mask=10, time_mask=50, *args, **kwargs):

        self.aug = Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask),
            torchaudio.transforms.TimeMasking(time_mask),
        )

    def __call__(self, data: torch.Tensor, **kwargs):
        return self.aug(data)
