import os.path
from typing import List, Tuple

import torch
import wget, gzip, shutil
from pyctcdecode import build_ctcdecoder

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:

        if isinstance(inds, torch.Tensor):
            inds = inds.tolist()

        result = ""
        last_char = None
        last_empty = False
        for ind in inds:
            if ind == 0:
                last_empty = True
                continue
            if len(result) == 0 or \
                    last_char != self.ind2char[ind] or \
                    (last_char == self.ind2char[ind] and last_empty):
                last_char = self.ind2char[ind]
                result += last_char
            last_empty = False
        return result
