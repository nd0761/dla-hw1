import logging
from typing import List
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    if len(dataset_items) < 1:
        return {}

    fields = ['audio', 'spectrogram', 'text_encoded']
    # fields = ['spectrogram', 'text_encoded']
    # fields_to_transfer = ['text']

    result_batch = {}

    for field in fields:
        list_items = []
        list_length = []
        for item in dataset_items:
            if field == fields[1]:
                list_items.append(torch.transpose(item[field], -1, -2).squeeze(0))
            else:
                list_items.append(item[field].squeeze(0))
            list_length.append(item[field].size()[-1])
        padded_tensors = pad_sequence(list_items, batch_first=True)
        result_batch[field] = padded_tensors
        result_batch[field + '_length'] = torch.Tensor(list_length).long()

    for item in dataset_items:
        for field in item.keys():
            if field in fields:
                continue
            if field not in result_batch.keys():
                result_batch[field] = []
            result_batch[field].append(item[field])
    return result_batch
