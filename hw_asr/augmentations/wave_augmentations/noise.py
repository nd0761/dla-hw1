from hw_asr.augmentations.base import AugmentationBase

import torch
import librosa


def load_noise(file_path):
    return file_path


class AddNoise(AugmentationBase):
    def __init__(self, bg_path='./data/bg_noise', coef=1e-3, normal_coef=0.05):
        # filename = librosa.ex(bg_path)
        # y, sr = librosa.load(filename)

        self.noise = "y"
        # self.coef = coef
        # self.normal_coef = normal_coef

    def __call__(self, data, **kwargs):
        return data
