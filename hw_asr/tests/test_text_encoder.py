import unittest

import numpy as np

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.text_encoder.ctc_char_bpe_encoder import CTCCharBpeEncoder
from hw_asr.text_encoder.char_text_encoder import CharTextEncoder

import youtokentome as yttm
EOS_IND = 3


def get_probs(given_text, encoder: CharTextEncoder):
    tokens = encoder.encode(given_text)[0]
    len_probas = tokens.shape[0]

    probas = np.zeros((len_probas, len(encoder.ind2char)))
    for ind, token in enumerate(tokens.tolist()):
        probas[ind, int(token)] = 1.0
    return probas


def union_given_probs(probas_a, probas_b):
    correct_size = max(probas_a.shape[0], probas_b.shape[0])
    probas_a = np.pad(probas_a, [(0, correct_size - probas_a.shape[0]), (0, 0)], mode='constant', )
    probas_b = np.pad(probas_b, [(0, correct_size - probas_b.shape[0]), (0, 0)], mode='constant')
    return probas_b + probas_a


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        text = "^i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_bpe_decode(self):
        bpe = CTCCharBpeEncoder()
        true_text = "i wish i started doing this hw earlier"
        encoded_text = bpe.encode(true_text)[0]
        decoded_text = bpe.ctc_decode(encoded_text)
        self.assertIn(decoded_text, true_text)

    def test_char_beam_search(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        text_encoder.init_kenlm()

        text1 = "i like to party"
        text2 = "i like to patty"
        probs_a = get_probs(text1, text_encoder)
        probs_b = get_probs(text2, text_encoder)
        probs = union_given_probs(0.51 * probs_a, 0.49 * probs_b)
        decoded_beams = text_encoder.ctc_beam_search(np.array(probs), beam_size=100)
        print(decoded_beams)
        self.assertIn(decoded_beams[0][0], text1)

    def test_bpe_beam_search(self):
        text_encoder = CTCCharBpeEncoder.get_simple_alphabet()
        text_encoder.init_kenlm(True)

        text1 = "he glared at me"
        text2 = "he glared to me"
        enc1 = text_encoder.encode(text1)[0]
        enc2 = text_encoder.encode(text2)[0]
        dec1 = text_encoder.ctc_decode(enc1)
        dec2 = text_encoder.ctc_decode(enc2)
        print('\n', dec1, dec2, sep='\n')
        print('\n', enc1, enc2, sep='\n')

        probs_a = get_probs(dec1, text_encoder)
        probs_b = get_probs(dec2, text_encoder)
        probs = union_given_probs(0.51 * probs_a, 0.49 * probs_b)
        print(len(text_encoder.alphabet))
        decoded_beams = text_encoder.ctc_beam_search(probs, beam_size=20)
        print(decoded_beams)
        # print(text_encoder.ctc_decoder.decode(probs))
        self.assertIn(decoded_beams[0][0], dec1)

