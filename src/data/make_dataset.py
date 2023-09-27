import requests
import pandas as pd
import os

class BaseDatasetMaker():
    def __init__(self, output_dir: str, file_path: str = None) -> None:
        if file_path == None:
            file_path = self.download_data()
        if not os.path.exists(file_path):
            raise Exception(f"Cannot find path {file_path}")
        # Set path to file with raw data
        self.file_path = file_path
        # Parse raw data into meaningful structure
        self.content = self.parse_data()
        # Extract toxic and detoxified texts as DataFrame
        extracted_text = self.extract_toxic_and_detoxified_text()
        # Save this DataFrame as csv file
        extracted_text.to_csv(output_dir)

    def download_data(self) -> str:
        raise Exception("Not implemented")

    def parse_data(self):
        raise Exception("Not implemented")
    
    def extract_toxic_and_detoxified_text(self) -> pd.DataFrame:
        raise Exception("Not implemented")

class ParaNMTDetoxDatasetMaker(BaseDatasetMaker):

    def download_data(self) -> str:
        url = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
        response = requests.get(url)
        data_file = open("filtered_paranmt.zip", "wb")
        data_file.write(response.content)

