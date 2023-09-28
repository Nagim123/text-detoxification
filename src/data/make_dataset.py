import requests
import pandas as pd
import os
import argparse
import logging
from zipfile import ZipFile

import pathlib
script_path = pathlib.Path(__file__).parent.resolve()
OUTPUT_DIR = os.path.join(script_path, "..\\..\\data\\interim")
INPUT_DIR = os.path.join(script_path, "..\\..\\data\\raw")


class BaseDatasetMaker():
    """
    Base class to make dataset from raw data
    """
    
    def __init__(self, output_dir: str, file_path: str = None, dataset_file_name: str = "uknown.csv") -> None:
        """
        Basic preprocessing pipeline.
        """
        self.output_dir = output_dir

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
        extracted_text.to_csv(os.path.join(output_dir, dataset_file_name))

    def download_data(self) -> str:
        raise Exception("Not implemented")

    def parse_data(self):
        raise Exception("Not implemented")
    
    def extract_toxic_and_detoxified_text(self) -> pd.DataFrame:
        raise Exception("Not implemented")

class ParaNMTDetoxDatasetMaker(BaseDatasetMaker):
    def __init__(self) -> None:
        file_path = os.path.join(INPUT_DIR, "filtered_paranmt.zip")
        super().__init__(OUTPUT_DIR, file_path, "filtered_paranmt.csv")

    def download_data(self) -> str:
        url = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
        file_path = os.path.join(INPUT_DIR, "filtered_paranmt.zip")
        
        response = requests.get(url)
        data_file = open(file_path, "wb")
        data_file.write(response.content)
    
    def parse_data(self):
        zip_object = ZipFile(self.file_path)
        zip_object.extractall(self.output_dir)
        zip_object.close()
        
        tsv_file_path = os.path.join(self.output_dir, "filtered.tsv")
        filtered_df = pd.read_csv(tsv_file_path, sep='\t')
        os.remove(tsv_file_path)
        return filtered_df
    
    def extract_toxic_and_detoxified_text(self) -> pd.DataFrame:
        parsed_df = self.content[['reference', 'translation']]
        parsed_df = parsed_df.rename(columns={'reference':"toxic", 'translation':"detoxified"})
        return parsed_df

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
