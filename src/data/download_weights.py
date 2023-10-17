import gdown
import os

from constants import WEIGHTS_DIR

def download_model_weights(url: str, output_dir: str):
    gdown.download(url, output_dir, quiet=False, fuzzy=True)

lstm_url = "https://drive.google.com/file/d/17kWTfP9WOcWKsK_0wXfYYzThbHdbK9lb/view?usp=share_link"
ae_lstm_url = "https://drive.google.com/file/d/1I4oFZWK7qecLEiTlPuD_euMn89WY_emp/view?usp=share_link"
transformer_url = "https://drive.google.com/file/d/1_EY9jxsJCnSleOaDqD1bT_jSC2DBct19/view?usp=share_link"

lstm_path = os.path.join(WEIGHTS_DIR, "lstm.pt")
ae_lstm_path = os.path.join(WEIGHTS_DIR, "ae_lstm.pt")
transformer_path = os.path.join(WEIGHTS_DIR, "transformer.pt")

if __name__ == "__main__":
    print("Start downloading model's weights...")
    download_model_weights(lstm_url, lstm_path)
    download_model_weights(ae_lstm_url, ae_lstm_path)
    download_model_weights(transformer_url, transformer_path)
    print("Download successful!")