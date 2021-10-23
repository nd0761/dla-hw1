from collections import Callable
from typing import List

import hw_asr.augmentations.spectrogram_augmentations
import hw_asr.augmentations.wave_augmentations
from hw_asr.augmentations.sequential import SequentialAugmentation
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.augmentations.random_apply import RandomApply


def from_configs(configs: ConfigParser):
    wave_augs = []
    p = configs.config["augmentations"]["random_apply"]
    if "augmentations" in configs.config and "wave" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["wave"]:
            wave_augs.append(
                RandomApply(
                    configs.init_obj(aug_dict, hw_asr.augmentations.wave_augmentations),
                    p
                )
            )

    spec_augs = []
    if "augmentations" in configs.config and "spectrogram" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["spectrogram"]:
            spec_augs.append(
                RandomApply(
                    configs.init_obj(aug_dict, hw_asr.augmentations.spectrogram_augmentations),
                    p
                )
            )
    return _to_function(wave_augs), _to_function(spec_augs)


def _to_function(augs_list: List[Callable]):
    if len(augs_list) == 0:
        return None
    elif len(augs_list) == 1:
        return augs_list[0]
    else:
        return SequentialAugmentation(augs_list)
