import logging
import argparse
import pandas as pd
import os
from zipfile import ZipFile
from constants import DATASET_PATH, OUTPUT_DIR_PATH, PREPROCESS_SCRIPT_PATH

def read_dataset() -> pd.DataFrame:
    """
    Read dataset from standard dataset path.

    Returns:
        DataFrame: Data frame of dataset.
    """

    # Unzip archieve
    zip_object = ZipFile(DATASET_PATH)
    zip_object.extractall(OUTPUT_DIR_PATH)
    zip_object.close()

    # Read data frame from .tsv
    tsv_file_path = os.path.join(OUTPUT_DIR_PATH, "filtered.tsv")
    dataset_df = pd.read_csv(tsv_file_path, sep='\t')
    
    # Delete content of archieve
    os.remove(tsv_file_path)

    return dataset_df

def parse_dataset(dataset_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Parse data frame to extract toxic and detoxified texts.
    
    Parameters:
        dataset_df (DataFrame): Data frame to parse.

    Returns:
        list[str]: List of toxic texts.
        list[str]: List of their detoxified versions.
    """

    # Create empty dataframe
    parsed_df = pd.DataFrame()

    # Set toxic column equal to rows from reference or translated columns depend on toxicity level (max).
    parsed_df["toxic"] = dataset_df.apply(lambda row: row["reference"].lower() if row["ref_tox"] > row["trn_tox"] else row["translation"].lower(), axis=1)
    # Set detoxified column equal to rows from reference or translated columns depend on toxicity level (min).
    parsed_df["detoxified"] = dataset_df.apply(lambda row: row["translation"].lower() if row['ref_tox'] > row['trn_tox'] else row['reference'].lower(), axis=1)
    # Set level of toxicity for detoxic texts 
    parsed_df['toxic_tox'] = dataset_df.apply(lambda row: max(row['ref_tox'], row['trn_tox']), axis=1)
    # Set level of toxicity for detoxified texts
    parsed_df['detoxified_tox'] = dataset_df.apply(lambda row: min(row['ref_tox'], row['trn_tox']), axis=1)
    
    # Remove rows with too low toxic text toxicity and too high detoxified text toxicity.
    rows_to_drop = parsed_df[(parsed_df['detoxified_tox'] > 0.2) | (parsed_df['toxic_tox'] < 0.8)]
    
    # Create new data frame with cleaned and filtered data
    balanced_df = parsed_df.drop(rows_to_drop.index)
    balanced_df.reset_index(inplace=True, drop=True)

    # Return only 'toxic' and 'detoxified' columns
    return balanced_df["toxic"].tolist(), balanced_df["detoxified"].tolist()



if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Prepare raw ParaNMT dataset for training.")
    parser.add_argument("--logging", action="store_true")
    args = parser.parse_args()

    # Set logging if required
    if args.logging:
        logging.basicConfig(level=logging.INFO)
    
    # Read dataset
    logging.info("Reading dataset...")
    df = read_dataset()
    # Parse dataset
    logging.info("Parsing dataset...")
    toxic, detoxified = parse_dataset(df)
    
    # Create temporal files to store texts
    temp1_path = os.path.join(OUTPUT_DIR_PATH, "temp1.txt")
    with open(temp1_path, "w", encoding="UTF-8") as temp:
        temp.write("\n".join(toxic))
    
    temp2_path = os.path.join(OUTPUT_DIR_PATH, "temp2.txt")
    with open(temp2_path, "w", encoding="UTF-8") as temp:
        temp.write("\n".join(detoxified))

    # Call external script to convert files to .pt file with tokenized and encoded data.
    logging.info("Tokenizing and saveing on disk...")
    do_logging = "--logging" if args.logging else ""
    os.system(f"python {PREPROCESS_SCRIPT_PATH} {temp1_path} dataset.pt --translated_text_file {temp2_path} --vocab_encode vocab.pt {do_logging}")
    # Remove temporal files
    os.remove(temp1_path)
    os.remove(temp2_path)
    logging.info("Done!")