from typing import List, Tuple

import torch
import os
import gdown
import youtokentome as yttm
from torch import Tensor

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharBpeEncoder(CharTextEncoder):

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        model_id = "1lZCyXmDeKcHc3GYF16aR23r3YS8T43_Y"
        model_path = [".", "bpe_models", "bpe.model"]
        model_path = os.path.join(*model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            gdown.download(id=model_id, output=model_path)

        self.bpe = yttm.BPE(model_path)

        alphabet = self.bpe.vocab()
        alphabet = alphabet[1:]
        super().__init__(alphabet)

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor(self.bpe.encode([text], output_type=yttm.OutputType.ID)).unsqueeze(0)
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
            if ind == self.char2ind[self.EMPTY_TOK]:
                last_empty = True
                continue
            if len(result) == 0 or \
                    last_char != self.ind2char[ind] or \
                    (last_char == self.ind2char[ind] and last_empty):
                last_char = self.ind2char[ind]
                result.append(last_char)
            last_empty = False
        return self.bpe.decode(result)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = []
        # TODO: your code here
        raise NotImplementedError
        return sorted(hypos, key=lambda x: x[1], reverse=True)
