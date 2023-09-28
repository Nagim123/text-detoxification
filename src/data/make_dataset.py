import requests
import pandas as pd
import os
import argparse
import logging

import pathlib
script_path = pathlib.Path(__file__).parent.resolve()

class BaseDatasetMaker():
    def __init__(self, output_dir: str, file_path: str = None) -> None:
        if file_path == None or not os.path.exists(file_path):
            logging.warning(f"Cannot find file {file_path}, downloading from internet...")
            file_path = self.download_data()
        
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
    def __init__(self) -> None:
        output_dir = os.path.join(script_path, "..\\..\\data\\interim")
        file_path = os.path.join(script_path, "..\\..\\data\\raw\\filtered_paranmt.zip")
        super().__init__(output_dir, file_path)

    def download_data(self) -> str:
        url = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
        file_path = os.path.join(script_path, "..\\..\\data\\raw\\filtered_paranmt.zip")
        
        response = requests.get(url)
        data_file = open(file_path, "wb")
        data_file.write(response.content)

supported_datasets = {
    "ParaNMT": ParaNMTDetoxDatasetMaker,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset", choices=["ParaNMT", "other"])
    args = parser.parse_args()

    if args.dataset in supported_datasets:
        supported_datasets[args.dataset]()
    else:
        raise Exception(f"Dataset {args.dataset} is not supported!")
