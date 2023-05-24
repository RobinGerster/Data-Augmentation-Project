# Text Augmentation for Classification (TAC) Library

This is a Python project currently focused on text classification tasks using transformer-based models such as BERT. It is intended to grow into a comprehensive library for experimenting with data augmentation techniques for text classification.

## Current Scripts

1. `data_loader.py`
2. `models.py`
3. `trainer.py`
4. `main.py`

### `data_loader.py`

This script provides functions and classes for loading text classification datasets.

**Features:**
1. `TextClassificationDataset`: A custom PyTorch Dataset class for a text classification problem. It supports text,label pairs.
2. `get_dataloader`: A function to load data from a CSV file and split it into training, validation, and test sets. CSV files must be in the format label,text

### `models.py`

This script includes the definition for the BERT sequence classification model.

**Features:**
1. `SequenceBertModel`: A class for the BERT model used for sequence classification. It uses a pretrained BERT model and tokenizes input sequences.

### `trainer.py`

This script includes the definition of a supervised trainer for training the model.

**Features:**
1. `SupervisedTrainer`: A class for training a sequence classification model. This includes training and evaluation methods.

### `main.py`

This script utilizes all of the above scripts to train a BERT model for sequence classification.

**Features:**
1. Loads a pretrained BERT model for sequence classification.
2. Loads data from a CSV file using the `get_dataloader` function.
3. Defines loss and optimizer.
4. Trains the model using `SupervisedTrainer`.
5. TO BE DONE: Implement History class to hold training information eg loss per epoch. The trainer should return this alongside a model!


