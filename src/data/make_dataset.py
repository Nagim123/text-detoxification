from zipfile import ZipFile
from tqdm import tqdm
import torch
import logging
import argparse
import spacy
import pandas as pd
import pathlib
import os

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
DATASET_PATH = os.path.join(SCRIPT_PATH, "../../data/raw/filtered_paranmt.zip")
OUTPUT_DIR_PATH = os.path.join(SCRIPT_PATH, "../../data/interim")

def read_dataset() -> pd.DataFrame:
    zip_object = ZipFile(DATASET_PATH)
    zip_object.extractall(OUTPUT_DIR_PATH)
    zip_object.close()

    tsv_file_path = os.path.join(OUTPUT_DIR_PATH, "filtered.tsv")
    dataset_df = pd.read_csv(tsv_file_path, sep='\t')
    os.remove(tsv_file_path)
    return dataset_df

def parse_dataset(dataset_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    parsed_df = pd.DataFrame()
    parsed_df["toxic"] = dataset_df.apply(lambda row: row["reference"].lower() if row["ref_tox"] > row["trn_tox"] else row["translation"].lower(), axis=1)
    parsed_df["detoxified"] = dataset_df.apply(lambda row: row["translation"].lower() if row['ref_tox'] > row['trn_tox'] else row['reference'].lower(), axis=1)
    parsed_df['detoxified_tox'] = dataset_df.apply(lambda row: min(row['ref_tox'], row['trn_tox']), axis=1)
    parsed_df['toxic_tox'] = dataset_df.apply(lambda row: max(row['ref_tox'], row['trn_tox']), axis=1)
    
    rows_to_drop = parsed_df[(parsed_df['detoxified_tox'] > 0.2) | (parsed_df['toxic_tox'] < 0.8)]
    balanced_df = parsed_df.drop(rows_to_drop.index)
    balanced_df.reset_index(inplace=True, drop=True)
    return balanced_df["toxic"].tolist(), balanced_df["detoxified"].tolist()

def tokenize_texts(texts: list[str], do_logging: bool = False):
    spacy_eng = spacy.load("en_core_web_sm")
    tokenized_texts = []
    if do_logging:
        texts = tqdm(texts)
    for text in texts:
        tokenized_texts.append([tok.text for tok in spacy_eng.tokenizer(text)])
    return tokenized_texts

def tokenize_text(text: list[str]):
    spacy_eng = spacy.load("en_core_web_sm")
    return [tok.text for tok in spacy_eng.tokenizer(text)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--logging", action="store_true")
    args = parser.parse_args()

    if args.logging:
        logging.basicConfig(level=logging.INFO)
    
    logging.info("Reading dataset...")
    df = read_dataset()
    logging.info("Parsing dataset...")
    toxic, detoxified = parse_dataset(df)
    
    logging.info("Tokenization for toxic text...")
    tokenized_toxic = tokenize_texts(toxic, args.logging)
    logging.info("Tokenization for detoxified text...")
    tokenized_detoxified = tokenize_texts(detoxified, args.logging)
    
    logging.info("Saving on disk...")
    torch.save({"toxic": tokenized_toxic, "detoxified": tokenized_detoxified}, os.path.join(OUTPUT_DIR_PATH, "dataset.pt"))
    logging.info("Done!")