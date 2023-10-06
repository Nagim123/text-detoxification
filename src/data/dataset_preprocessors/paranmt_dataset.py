from dataset_preprocessors.basic_dataset import BaseDatasetMaker
import requests
import os
import pandas as pd
from zipfile import ZipFile

class ParaNMTDetoxDatasetMaker(BaseDatasetMaker):
    """
    Dataset maker for ParaNMT dataset.
    """
    
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer, "filtered_paranmt.zip", "filtered_paranmt.csv")

    def download_data(self) -> None:
        # Url to download ParaNMT
        url = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
        
        # Download from url and save to file
        response = requests.get(url)
        data_file = open(self.file_path, "wb")
        data_file.write(response.content)
    
    def parse_data(self):

        # Unzip archive with dataset
        zip_object = ZipFile(self.file_path)
        zip_object.extractall(BaseDatasetMaker.output_dir)
        zip_object.close()
        
        # Read .tsv file that was in archieve
        tsv_file_path = os.path.join(BaseDatasetMaker.output_dir, "filtered.tsv")
        filtered_df = pd.read_csv(tsv_file_path, sep='\t')
        os.remove(tsv_file_path)
        return filtered_df
    
    def extract_toxic_and_detoxified_text(self) -> pd.DataFrame:
        parsed_df = pd.DataFrame()
        # Seperate translated and reference texts into toxic and detoxified ones
        # Also apply lowercasing
        parsed_df["toxic"] = self.content.apply(lambda row: row["reference"].lower() if row["ref_tox"] > row["trn_tox"] else row["translation"].lower(), axis=1)
        parsed_df["detoxified"] = self.content.apply(lambda row: row["translation"].lower() if row['ref_tox'] > row['trn_tox'] else row['reference'].lower(), axis=1)
        return parsed_df["toxic"].tolist(), parsed_df["detoxified"].tolist()
    
    def tokenize_data(self, toxic_text: list, detoxified_text: list) -> pd.DataFrame:
        full_text = toxic_text + detoxified_text
        self.tokenizer.create_vocab(full_text)
        result_dataframe = pd.DataFrame()
        result_dataframe["input"] = [self.tokenizer.tokenize(text) for text in toxic_text]
        result_dataframe["label"] = [self.tokenizer.tokenize(text) for text in detoxified_text]
        result_dataframe["vocab_size"] = len(self.tokenizer)
        return result_dataframe