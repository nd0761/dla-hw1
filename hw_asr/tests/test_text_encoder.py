import unittest

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.text_encoder.ctc_char_bpe_encoder import CTCCharBpeEncoder

import youtokentome as yttm


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
        encoded_text = bpe.encode(true_text)
        print('\n', len(bpe.vocab))
        decoded_text = bpe.ctc_decode(encoded_text)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()

