from typing import List, Tuple

import torch
import os
import gdown
import youtokentome as yttm
from torch import Tensor

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharBpeEncoder(CharTextEncoder):

    def __init__(self, **kwargs):
        self.EMPTY_TOK = '_'
        model_id = "1igehfoD46XM8cmAxK4oLF1Czh_pE7"
        model_path = [".", "bpe_models", "bpe.model"]
        model_path = os.path.join(*model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            gdown.download(id=model_id, output=model_path)

        self.bpe = yttm.BPE(model_path)

        alphabet = self.bpe.vocab()
        super().__init__(alphabet)

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor(self.bpe.encode([text], output_type=yttm.OutputType.ID))
        except KeyError as e:
            raise Exception(
                f"Can't encode text '{text}'")

    def ctc_decode(self, inds: List[int]) -> str:
        if isinstance(inds, torch.Tensor):
            inds = inds.tolist()

        result = []
        last_char = None
        last_empty = False
        for ind in inds:
            new_char = self.ind2char[ind]
            if ind == 0:
                last_empty = True
                continue
            if len(result) == 0 or \
                    last_char != new_char or \
                    (last_char == new_char and last_empty):
                last_char = new_char
                result.append(ind)
            last_empty = False
        return self.bpe.decode([result])[0]
