import argparse
import collections
import warnings

import numpy as np
import torch
import os
import gdown
import shutil
import json
import youtokentome as yttm

import hw_asr.loss as module_loss
import hw_asr.metric as module_metric
import hw_asr.model as module_arch
from hw_asr.datasets.utils import get_dataloaders
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.trainer import Trainer
from hw_asr.utils import prepare_device
from hw_asr.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def load_data(folder_path, drive_id):
    if os.path.exists(folder_path):
        return
    os.makedirs(folder_path, exist_ok=True)
    file_name = "train-clean-100_index.json"
    gdown.download(id=drive_id, output=os.path.join(folder_path, file_name))


def json_to_str(librispeech_json_path, txt_whole_path):
    if not os.path.exists(librispeech_json_path):
        raise Exception("Path to data doesn't exist")
    with open(librispeech_json_path) as libri_file:
        data = json.load(libri_file)
        all_texts = []
        for datum in data:
            all_texts.append(datum["text"])
        joined_text = '\n'.join(all_texts)
        with open(txt_whole_path, "w+") as text_file:
            text_file.write(joined_text)
        return txt_whole_path


def main(config):
    model_bpe_list = config.config["bpe"]["model_path"]
    model_bpe_path = os.path.join(*model_bpe_list)
    if not os.path.exists(model_bpe_path):
        os.makedirs(model_bpe_path)
    print("BPE folder:", model_bpe_path)

    data_list = config.config["bpe"]["data_path"]
    data_path = os.path.join(*data_list)

    text_json_list = config.config["bpe"]["json_path"]
    text_json_path = os.path.join(*text_json_list)
    print("path to text data as json:", text_json_path)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    txt_name = "all_texts.txt"
    all_texts = os.path.join(data_path, txt_name)
    if not os.path.exists(all_texts):
        all_texts = json_to_str(text_json_path, all_texts)
    print("path to text data as txt:", all_texts)

    model_name = "bpe.model"

    yttm.BPE.train(data=all_texts, vocab_size=5000, model=os.path.join(model_bpe_path, model_name))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
