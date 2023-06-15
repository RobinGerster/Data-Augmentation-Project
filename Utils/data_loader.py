import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        if self.labels:
            label = torch.tensor(self.labels[idx])
            return text, label
        else:
            return text


def get_dataloader(csv_path, splits=[1.0], batch_sizes=[16]):
    # Read data
    df = pd.read_csv(csv_path, header=None, names=["label", "text"])

    # Verify splits sum to 1
    assert sum(splits) == 1, "Splits must sum to 1"

    # Verify the number of batch sizes matches the number of splits
    assert len(splits) == len(batch_sizes), "There should be the same number of batch_sizes as splits"

    # Split data
    remain = df
    datasets = []
    dataloaders = []

    for split in splits[:-1]:
        train, remain = train_test_split(remain, test_size=1 - split, random_state=42)
        datasets.append(train)

    datasets.append(remain)

    for idx, dataset in enumerate(datasets):
        texts = dataset["text"].tolist()
        labels = dataset["label"].tolist()

        dataset = TextClassificationDataset(texts, labels)
        dataloader = DataLoader(dataset, batch_size=batch_sizes[idx], shuffle=True)

        dataloaders.append(dataloader)

    return dataloaders


def get_ssmba_dataloaders(batch_sizes, naug=1, bias=False, dataset="IMDB"):
    if dataset == "IMDB":
        if not bias:
            train_df = pd.read_csv("../Datasets/imdb_no_bias/" + dataset + "_no_bias_" + str(naug) + "_ssmba_train.csv", header=None, names=["label", "text"])
            test_df = pd.read_csv("../Datasets/" + dataset + "_1000_ssmba_val.csv", header=None, names=["label", "text"])
            ood_df = pd.read_csv("../Datasets/SST-2_1000_ssmba_test.csv", header=None, names=["label", "text"])
        else:
            train_df = pd.read_csv("../Datasets/imdb_bias/" + dataset + "_bias_" + str(naug) + "_ssmba_train.csv", header=None, names=["label", "text"])
            test_df = pd.read_csv("../Datasets/" + dataset + "_1000_ssmba_bias_val.csv", header=None, names=["label", "text"])
            ood_df = pd.read_csv("../Datasets/SST-2_1000_ssmba_bias_test.csv", header=None, names=["label", "text"])
    if dataset == "MNLI":
        if not bias:
            train_df = pd.read_csv("../Datasets/mnli_no_bias/" + dataset + "_no_bias_" + str(naug) + "_ssmba_train.csv", header=None, names=["label", "text"])
            test_df = pd.read_csv("../Datasets/" + dataset + "_ssmba_val.csv", header=None, names=["label", "text"])
            ood_df = pd.read_csv("../Datasets/" + dataset + "_ssmba_test.csv", header=None, names=["label", "text"])
        else:
            train_df = pd.read_csv("../Datasets/mnli_bias/" + dataset + "_bias_" + str(naug) + "_ssmba_train.csv", header=None, names=["label", "text"])
            test_df = pd.read_csv("../Datasets/" + dataset + "_ssmba_bias_val.csv", header=None, names=["label", "text"])
            ood_df = pd.read_csv("../Datasets/" + dataset + "_ssmba_bias_test.csv", header=None, names=["label", "text"])

    datasets = [train_df, test_df, ood_df]
    loaders = []

    for i, data in enumerate(datasets):
        texts = data["text"].tolist()
        labels = data["label"].tolist()

        data = TextClassificationDataset(texts, labels)
        dataloader = DataLoader(data, batch_size=batch_sizes[i], shuffle=True)

        loaders.append(dataloader)

    return loaders
