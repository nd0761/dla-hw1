from typing import List, Tuple

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharBpeEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
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

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.numpy()
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = []
        # TODO: your code here
        raise NotImplementedError
        return sorted(hypos, key=lambda x: x[1], reverse=True)
