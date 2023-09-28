import argparse
from dataset_preprocessors.paranmt_dataset import ParaNMTDetoxDatasetMaker

supported_datasets = {
    "ParaNMT": ParaNMTDetoxDatasetMaker,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset", choices=list(supported_datasets.keys()))
    args = parser.parse_args()

    if args.dataset in supported_datasets:
        supported_datasets[args.dataset]()
    else:
        raise Exception(f"Dataset {args.dataset} is not supported!")
