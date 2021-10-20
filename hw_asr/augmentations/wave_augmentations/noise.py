import random

from hw_asr.augmentations.base import AugmentationBase

import torch
import librosa
import gdown
import os
import shutil
import random
import torchaudio


def load_noise(file_path, drive_id):
    if os.path.exists(file_path):
        return
    os.makedirs(file_path)
    gdown.download(id=drive_id, output=os.path.join(file_path, "noise.zip"))
    shutil.unpack_archive(os.path.join(file_path, "noise.zip"), file_path, "zip")
    os.remove(os.path.join(file_path, "noise.zip"))


class AddNoise(AugmentationBase):
    def __init__(self, bg_path='./data/bg_noise',
                 drive_id="18l1uBmLoAYAFzqDFqdRXmfbEciz4QFjG",
                 sr=16000, max_noise=8, min_noise=0,
                 *args, **kwargs):
        bg_path = os.path.join(".", "data", "bg_noise") # vindovs sosatb
        print("-----augmentation_with_noise")
        load_noise(bg_path, drive_id)
        print("-----loaded_noise")
        noise_folder_path = os.path.join(bg_path, "bg")
        self.noises = [f for f in os.listdir(noise_folder_path)
                       if os.path.isfile(os.path.join(noise_folder_path, f))]
        self.max_level = max_noise
        self.min_level = min_noise
        self.bg_path = noise_folder_path
        self.sr = sr

    def __call__(self, data, **kwargs):
        print(gdown.__version__)
        noise_name = random.choice(self.noises)
        noise_path = os.path.join(self.bg_path, noise_name)

        noise, noise_sr = torchaudio.load(noise_path)
        noise = noise[:1, :]
        if self.sr != noise_sr:
            noise = torchaudio.functional.resample(noise, noise_sr, self.sr)

        noise = noise.squeeze(0)
        data = data.squeeze(0)

        noise_level = torch.Tensor([random.uniform(self.min_level, self.max_level)])

        max_noise_size = 0.5 * data.shape[0]
        noise_beginning = random.randint(0, int(max_noise_size))
        noise_len = data.shape[0] - noise_beginning
        clipped_noise = noise[:noise_len]

        noise_energy = torch.norm(clipped_noise)
        audio_energy = torch.norm(data)
        alpha = (audio_energy / noise_energy) * torch.pow(10, -noise_level / 20)
        data[noise_beginning:noise_beginning+noise_len] += alpha * clipped_noise
        return torch.clamp(data, -1, 1)
