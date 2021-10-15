import numpy as np
import editdistance
# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    if target_text == '' and predicted_text == '':
        return 0.
    elif target_text == '':
        return 1.

    return float(editdistance.eval(target_text, predicted_text)) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if len(target_text.split(' ')) == 0 and len(predicted_text.split(' ')) == 0:
        return 0.
    elif len(target_text.split(' ')) == 0:
        return 1.

    return float(editdistance.eval(target_text.split(' '), predicted_text.split(' '))) / len(target_text.split(' '))
