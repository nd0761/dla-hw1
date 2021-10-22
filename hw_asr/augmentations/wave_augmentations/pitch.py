import torch

from hw_asr.augmentations.base import AugmentationBase
from torchaudio.transforms import Vol


class AddPitchShifting(AugmentationBase):
    def __init__(self, min_shift=0.2, max_shift=0.5, **kwargs):
        self.min_shift = min_shift
        self.max_shift = max_shift

    def __call__(self, data, **kwargs):
        temp = (self.max_shift - self.min_shift) * torch.rand(1) + self.min_shift
        temp = temp.item()
        return Vol(temp)(data)
