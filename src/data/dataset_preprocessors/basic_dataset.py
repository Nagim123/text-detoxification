import pandas as pd
import logging
import os

import pathlib
script_path = pathlib.Path(__file__).parent.resolve()

class BaseDatasetMaker():
    """
    Base class to make dataset from raw data
    """
    
    output_dir = os.path.join(script_path, "..\\..\\..\\data\\interim")
    input_dir = os.path.join(script_path, "..\\..\\..\\data\\raw")

    def __init__(self, file_name: str = None, result_file_name: str = "uknown.csv") -> None:
        """
        Basic preprocessing pipeline.
        """
        # Set path to file with raw data
        self.file_path = os.path.join(BaseDatasetMaker.input_dir, file_name)
        # Set path to file with preprocess results
        self.result_file_path = os.path.join(BaseDatasetMaker.output_dir, result_file_name)

        if os.path.exists(self.result_file_path):
            logging.warning(f"{result_file_name} already exists. Deleting it...")
            os.remove(self.result_file_path)

        if not os.path.exists(self.file_path):
            logging.warning(f"Cannot find file {self.file_path}, downloading from internet...")
            self.download_data()
        
        # Parse raw data into meaningful structure
        self.content = self.parse_data()
        # Extract toxic and detoxified texts as DataFrame
        extracted_text = self.extract_toxic_and_detoxified_text()
        # Save this DataFrame as csv file
        extracted_text.to_csv(self.result_file_path)

    def download_data(self) -> None:
        raise Exception("Not implemented")

    def parse_data(self):
        raise Exception("Not implemented")
    
    def extract_toxic_and_detoxified_text(self) -> pd.DataFrame:
        raise Exception("Not implemented")