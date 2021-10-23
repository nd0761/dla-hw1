import os.path
from typing import List, Tuple

import torch
import wget, gzip, shutil
from pyctcdecode import build_ctcdecoder

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


# beam search from -> https://github.com/kensho-technologies/pyctcdecode/blob/main/tutorials/01_pipeline_nemo.ipynb


class CTCCharTextEncoder(CharTextEncoder):

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        self.alphabet = alphabet
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def init_kenlm(self):
        kenlm_path, unigram_list = self.prepare_kenlm()
        self.ctc_decoder = build_ctcdecoder(
            [""] + self.alphabet,
            kenlm_path,
            unigram_list
        )

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

    def prepare_kenlm(self):
        gz_three_gram_path = "3-gram.pruned.1e-7.arpa.gz"
        data_dir = os.path.join(*[".", "data", "librispeech_beam"])
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        gz_path = os.path.join(data_dir, gz_three_gram_path)
        if not os.path.exists(gz_path):
            gz_path = wget.download(
                'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz',
                out=data_dir
            )

        upper_lm_path = gz_three_gram_path[:-3]
        if not os.path.exists(os.path.join(*[data_dir, upper_lm_path])):
            with gzip.open(gz_path, 'rb') as f_zipped:
                with open(os.path.join(data_dir, upper_lm_path), 'wb') as f_unzipped:
                    shutil.copyfileobj(f_zipped, f_unzipped)

        lm_path = 'lowercase_3-gram.pruned.1e-7.arpa'
        if not os.path.exists(os.path.join(*[data_dir, lm_path])):
            with open(os.path.join(*[data_dir, upper_lm_path]), 'r') as f_upper:
                with open(os.path.join(*[data_dir, lm_path]), 'w') as f_lower:
                    for line in f_upper:
                        f_lower.write(line.lower().replace("'", ""))

        lib_vocab = "librispeech-vocab.txt"
        if not os.path.exists(os.path.join(*[data_dir, lib_vocab])):
            ls_url = "http://www.openslr.org/resources/11/librispeech-vocab.txt"
            wget.download(ls_url, out=str(data_dir))

        with open(os.path.join(*[data_dir, lib_vocab])) as f:
            unigram_list = [t.lower().replace("'", "") for t in f.read().strip().split("\n")]

        return str(os.path.join(*[data_dir, lm_path])), unigram_list

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().numpy()

        hypos = self.ctc_decoder.decode_beams(probs, beam_width=beam_size, token_min_logp=-13.)

        return [(x[0], x[4]) for x in hypos]
