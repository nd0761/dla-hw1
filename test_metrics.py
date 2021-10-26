import json

from hw_asr.metric.cer_metric import calc_cer
from hw_asr.metric.wer_metric import calc_wer
OUTPUT_PATH = "output.json"


def run_main_worker():
    with open(OUTPUT_PATH, "r") as json_file:
        output = json.load(json_file)
        print(output[0]["ground_trurh"])

        number_of_items = 0
        average_wer_argmax = 0.0
        average_cer_argmax = 0.0
        average_wer_bs = 0.0
        average_cer_bs = 0.0

        for line in output:
            number_of_items += 1
            ground_truth = line["ground_trurh"]
            argmax_res = line["pred_text_argmax"]
            best_bs = line["pred_text_beam_search"][0][0]

            average_wer_argmax += calc_wer(argmax_res, ground_truth)
            average_wer_bs += calc_wer(argmax_res, best_bs)

            average_cer_argmax += calc_cer(argmax_res, ground_truth)
            average_cer_bs += calc_cer(argmax_res, best_bs)

        average_wer_argmax /= number_of_items
        average_cer_argmax /= number_of_items
        average_wer_bs /= number_of_items
        average_cer_bs /= number_of_items

        print("\nWER argmax:", average_wer_argmax,
              "\nCER argmax:", average_cer_argmax,
              "\nWER beam search:", average_wer_bs,
              "\nCER beam search:", average_cer_bs,
              )
        with open("test_res.txt", "w") as results:
            results.write("\nWER argmax:" + str(average_wer_argmax))
            results.write("\nCER argmax:" + str(average_cer_argmax))
            results.write("\nWER beam search:" + str(average_wer_bs))
            results.write("\nCER beam search:" + str(average_cer_bs))


if __name__ == "__main__":
    run_main_worker()