"""
All file paths and vocabulary related constants.
"""

import pathlib
import os

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
DATASET_PATH = os.path.join(SCRIPT_PATH, "../../data/raw/filtered_paranmt.zip")
OUTPUT_DIR_PATH = os.path.join(SCRIPT_PATH, "../../data/interim")
PREPROCESS_SCRIPT_PATH = os.path.join(SCRIPT_PATH, "preprocess_texts.py")
VOCAB_SIZE = 10_000
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']
