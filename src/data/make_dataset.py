import logging
import argparse
import pandas as pd
import os

from zipfile import ZipFile
from constants import DATASET_PATH, OUTPUT_DIR_PATH, PREPROCESS_SCRIPT_PATH

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
    
    temp1_path = os.path.join(OUTPUT_DIR_PATH, "temp1.txt")
    temp2_path = os.path.join(OUTPUT_DIR_PATH, "temp2.txt")

    with open(temp1_path, "w", encoding="UTF-8") as temp:
        temp.write("\n".join(toxic))
    
    with open(temp2_path, "w", encoding="UTF-8") as temp:
        temp.write("\n".join(detoxified))

    logging.info("Tokenizing and saveing on disk...")
    do_logging = "--logging" if args.logging else ""
    os.system(f"python {PREPROCESS_SCRIPT_PATH} {temp1_path} dataset.pt --tranlated_text_file {temp2_path} --vocab_encode vocab.pt {do_logging}")
    os.remove(temp1_path)
    os.remove(temp2_path)
    logging.info("Done!")