from Utils.classifiers import SequenceBertClassifier
import pandas as pd
import torch

# The vector embedding associated to each text is simply the hidden state that Bert outputs for the [CLS] token.

device = "cpu"
bert = SequenceBertClassifier(device, pretrained_model_name="bert-base-uncased", num_labels=2)
tokenizer = bert.tokenizer

train_df = pd.read_csv("../Datasets/IMDB_500_4_ssmba_train.csv", header=None, names=["label", "text"])
tokenized_train = tokenizer(train_df["text"].values.tolist(), padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
  hidden_train = bert(**tokenized_train)

# Extract the vector embeddings
train_embeddings = hidden_train.last_hidden_state[:, 0, :]
