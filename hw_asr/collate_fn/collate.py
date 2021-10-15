import logging
from typing import List
import torch
import numpy as np

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    if len(dataset_items) < 2:
        return_batch = {}
        for item in dataset_items:
            for field in item.keys():
                if field not in return_batch:
                    return_batch[field] = []
                return_batch[field].append(item[field])
        return dataset_items

    max_len_fields = dict()
    fields = ['audio', 'spectrogram', 'text_encoded']

    new_field = ['text_encoded_length', 'spectrogram_length']
    text_lengths = []
    for item in dataset_items:
        text_lengths.append(item[fields[2]].size()[-1])
    spectrogram_length = []
    for item in dataset_items:
        spectrogram_length.append(item[fields[1]].size()[-1])

    for field in fields:
        max_len = 0
        for item in dataset_items:
            max_len = max(max_len, item[field].size()[-1])
        max_len_fields[field] = max_len

    result_batch = {new_field[0]: torch.tensor(text_lengths).long(),
                    new_field[1]: torch.tensor(spectrogram_length).long()}

    for item in dataset_items:
        for field in item.keys():
            value = item[field]
            if field in fields:
                pad1 = (max_len_fields[field] - value.size()[-1]) // 2
                pad2 = max_len_fields[field] - value.size()[-1] - pad1
                value = torch.nn.functional.pad(value, (pad1, pad2))
            if field not in result_batch:
                result_batch[field] = []
            result_batch[field].append(value)

    for field in fields:
        result_batch[field] = torch.stack(result_batch[field])
        result_batch[field] = result_batch[field].squeeze(1)
    result_batch['spectrogram'] = torch.transpose(result_batch['spectrogram'], 1, 2)
    return result_batch
