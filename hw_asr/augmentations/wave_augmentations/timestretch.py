import torch

from hw_asr.augmentations.base import AugmentationBase
import librosa


class AddTimeStretch(AugmentationBase):
    def __init__(self, min_stretch=0.8, max_stretch=1.2, *args, **kwargs):
        self.min_stretch = min_stretch
        self.max_stretch = max_stretch

    def __call__(self, data, **kwargs):
        temp = self.min_stretch + (self.max_stretch - self.min_stretch) * torch.rand(1)
        temp = temp.item()
        return torch.from_numpy(librosa.effects.time_stretch(data.numpy().squeeze(), temp))
