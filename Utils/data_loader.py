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


def get_imdb_training_test_data(batch_sizes, save=False):
    # Specify there is no header in the file
    df = pd.read_csv("../Datasets/IMDB_Full.csv", header=None, names=["label", "text"])

    # Get 2k as validation set
    val_df = df.sample(frac=0.2)

    # Create the remaining data as the train set with 40k
    train_df = df.drop(val_df.index)

    datasets = [train_df, val_df]
    loaders = []

    # Save these examples in a separate file for ssmba augmentation
    if save:
        train_df.to_csv('../Datasets/IMDB_20k_ssmba_train.csv', index=False, header=False)
        val_df.to_csv('../Datasets/IMDB_5k_ssmba_test.csv', index=False, header=False)

    for idx, dataset in enumerate(datasets):
        texts = dataset["text"].tolist()
        labels = dataset["label"].tolist()

        dataset = TextClassificationDataset(texts, labels)
        dataloader = DataLoader(dataset, batch_size=batch_sizes[idx], shuffle=True)

        loaders.append(dataloader)

    return loaders


if __name__ == "__main__":
    get_imdb_training_test_data([2, 2])
