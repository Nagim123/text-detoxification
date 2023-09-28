from dataset_preprocessors.basic_dataset import BaseDatasetMaker
import requests
import os
import pandas as pd
from zipfile import ZipFile

class ParaNMTDetoxDatasetMaker(BaseDatasetMaker):
    def __init__(self) -> None:
        super().__init__("filtered_paranmt.zip", "filtered_paranmt.csv")

    def download_data(self) -> None:
        url = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
        
        response = requests.get(url)
        data_file = open(self.file_path, "wb")
        data_file.write(response.content)
    
    def parse_data(self):
        zip_object = ZipFile(self.file_path)
        zip_object.extractall(BaseDatasetMaker.output_dir)
        zip_object.close()
        
        tsv_file_path = os.path.join(BaseDatasetMaker.output_dir, "filtered.tsv")
        filtered_df = pd.read_csv(tsv_file_path, sep='\t')
        os.remove(tsv_file_path)
        return filtered_df
    
    def extract_toxic_and_detoxified_text(self) -> pd.DataFrame:
        parsed_df = self.content[['reference', 'translation']]
        parsed_df = parsed_df.rename(columns={'reference':"toxic", 'translation':"detoxified"})
        return parsed_df