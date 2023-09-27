import requests
import pandas as pd

class BaseDatasetMaker():
    def __init__(self, output_dir: str, file_path: str = None) -> None:
        if file_path == None:
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
        pass

    def parse_data(self):
        pass
    def extract_toxic_and_detoxified_text(self) -> pd.DataFrame:
        pass

class ParaNMTDetoxDatasetMaker(BaseDatasetMaker):
    pass