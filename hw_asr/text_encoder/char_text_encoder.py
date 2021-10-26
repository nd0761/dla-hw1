import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union, Tuple
import torch

import numpy as np
from torch import Tensor
from pyctcdecode import build_ctcdecoder
import os, gzip, shutil, wget

from hw_asr.base.base_text_encoder import BaseTextEncoder
# beam search from -> https://github.com/kensho-technologies/pyctcdecode/blob/main/tutorials/01_pipeline_nemo.ipynb


class CharTextEncoder(BaseTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        self.alphabet = alphabet
        self.ind2char = {k: v for k, v in enumerate(alphabet)}
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.ctc_decoder = None

    def __len__(self):
        return len(self.ind2char)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError as e:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'")

    def init_kenlm(self, set_alphabet=False):
        alphabet = self.alphabet
        if not set_alphabet:
            alphabet = [""] + alphabet
        kenlm_path, unigram_list = self.prepare_kenlm()
        self.ctc_decoder = build_ctcdecoder(
            alphabet,
            kenlm_path,
            unigram_list
        )

    def prepare_kenlm(self):
        gz_three_gram_path = "3-gram.pruned.1e-7.arpa.gz"
        data_dir = os.path.join(*[".", "data", "librispeech_beam"])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
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

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return ''.join([self.ind2char[int(ind)] for ind in vector]).strip()

    def dump(self, file):
        with Path(file).open('w') as f:
            json.dump(self.ind2char, f)

    @classmethod
    def from_file(cls, file):
        with Path(file).open() as f:
            ind2char = json.load(f)
        a = cls([])
        a.ind2char = ind2char
        a.char2ind = {v: k for k, v in ind2char}
        return a

    @classmethod
    def get_simple_alphabet(cls):
        return cls(alphabet=list(ascii_lowercase + ' '))

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100,  **kwargs) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2

        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().numpy()

        hypos = self.ctc_decoder.decode_beams(probs, beam_width=beam_size, token_min_logp=-20.)

        return [(x[0], x[4]) for x in hypos]
