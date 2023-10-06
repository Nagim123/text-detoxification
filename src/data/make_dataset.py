import argparse
from dataset_preprocessors.paranmt_dataset import ParaNMTDetoxDatasetMaker
from text_tokenizers.custom_torch_tokenizer import CustomTorchTokenizer
from text_tokenizers.bpe_tokenizer import BPETokenizer

supported_datasets = {
    "ParaNMT": ParaNMTDetoxDatasetMaker,
}

supported_tokenizers = {
    "CustomTokenizer": CustomTorchTokenizer(),
    "BPETokenizer": BPETokenizer(),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset", choices=list(supported_datasets.keys()))
    parser.add_argument("tokenizer", choices=list(supported_tokenizers.keys()))
    args = parser.parse_args()

    if args.dataset in supported_datasets:
        supported_datasets[args.dataset](supported_tokenizers[args.tokenizer])
    else:
        raise Exception(f"Dataset {args.dataset} is not supported!")
