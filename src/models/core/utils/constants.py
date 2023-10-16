import pathlib
import os

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
DATASET_PATH = os.path.join(SCRIPT_PATH, "../../../../data/interim/dataset.pt")
MODEL_WEIGHTS_PATH = os.path.join(SCRIPT_PATH, "../../../../models/weights.pt")
MAX_SENTENCE_SIZE = 100
VOCAB_SIZE = 10_000
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3