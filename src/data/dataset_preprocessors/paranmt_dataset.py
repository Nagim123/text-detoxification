from dataset_preprocessors.basic_dataset import BaseDatasetMaker
import requests
import os
import pandas as pd
from zipfile import ZipFile

class ParaNMTDetoxDatasetMaker(BaseDatasetMaker):
    """
    Dataset maker for ParaNMT dataset.
    """
    
    def __init__(self) -> None:
        super().__init__("filtered_paranmt.zip", "filtered_paranmt.csv")

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
        parsed_df["toxic"] = self.content.apply(lambda row: row["reference"] if row["ref_tox"] > row["trn_tox"] else row["translation"], axis=1)
        parsed_df["detoxified"] = self.content.apply(lambda row: row["translation"] if row['ref_tox'] > row['trn_tox'] else row['reference'], axis=1)
        return parsed_df