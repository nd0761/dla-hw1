import torch

from hw_asr.augmentations.base import AugmentationBase
import librosa


class AddTimeStretch(AugmentationBase):
    def __init__(self, min_stretch=0.5, max_stretch=1.3, *args, **kwargs):
        self.min_stretch = min_stretch
        self.max_stretch = max_stretch

    def __call__(self, data, **kwargs):
        stretch_coef = torch.rand(1)
        temp = self.min_stretch + (self.max_stretch - self.min_stretch) * stretch_coef
        temp = temp.item()
        if isinstance(data, torch.Tensor):
            data = data.numpy().squeeze()
        return torch.from_numpy(librosa.effects.time_stretch(data, temp))
