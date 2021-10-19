from hw_asr.augmentations.base import AugmentationBase

import torch
import librosa


class AddNoise(AugmentationBase):
    def __init__(self, noise_type='trumpet', coef=1e-3, normal_coef=0.05):
        filename = librosa.ex(noise_type)
        y, sr = librosa.load(filename)

        self.noise = y
        self.coef = coef
        self.normal_coef = normal_coef

    def __call__(self, wav):
        # noise_level = torch.Tensor([20])  # [0, 40]

        noise_level = torch.norm(torch.from_numpy(self.noise))
        audio_energy = torch.norm(wav)

        alpha = (audio_energy / noise_level) * torch.pow(10, -noise_level / 20)

        clipped_wav = wav[..., :self.noise.shape[0]]

        augumented_wav = clipped_wav + alpha * torch.from_numpy(self.noise)

        # In some cases the resulting sound may go beyond [-1, 1]
        # So, clamp it :)
        augumented_wav = torch.clamp(augumented_wav, -1, 1)

        # noiser = distributions.Normal(0, self.normal_coef)
        return augumented_wav
