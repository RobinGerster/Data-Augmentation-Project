# Text Augmentation for Classification (TAC) Library

This is a Python project focused on text classification tasks using transformer-based models such as BERT. The goal of this library is to provide a comprehensive set of tools and techniques for data augmentation in text classification.

## Current Scripts

1. `data_loader.py`: Provides functions and classes for loading text classification datasets.

   **Features:**
   - `TextClassificationDataset`: A custom PyTorch Dataset class for text classification problems that supports text-label pairs.
   - `get_dataloader`: A function to load data from a CSV file and split it into training, validation, and test sets. CSV files must be in the format `label,text`.

2. `classifiers.py`: Contains the definition for the BERT sequence classification model.

   **Features:**
   - `SequenceBertModel`: A class representing the BERT model used for sequence classification. It utilizes a pretrained BERT model and tokenizes input sequences.

3. `trainer.py`: Includes the definition of supervised trainers for training the model.

   **Features:**
   - `SupervisedTrainer`: A class for training a sequence classification model. It provides methods for training and evaluation.
   - `UDATrainer`: A class for training a sequence classification model using Unsupervised Data Augmentation (UDA) techniques. It extends the functionality of `SupervisedTrainer` and incorporates unsupervised data augmentation into the training process.

4. `main.py`: Utilizes the above scripts to train a BERT model for sequence classification.

   **Features:**
   - Loads a pretrained BERT model for sequence classification.
   - Loads data from a CSV file using the `get_dataloader` function.
   - Defines loss and optimizer.
   - Trains the model using the appropriate trainer (`SupervisedTrainer` or `UDATrainer`).

5. `augment.py`: Contains different data augmentation approaches for text classification.

## Next Steps

The next step for the TAC Library is to complete the implementation of the `History` class, which will store training information such as loss per epoch. The trainers (`SupervisedTrainer` and `UDATrainer`) will be modified to return both the trained model and the training history for analysis and evaluation.

Additionally, the `augment.py` script will be expanded to include a variety of data augmentation approaches for text classification tasks, providing users with a wide range of options to enhance their text classification models.

Please let me know if there is anything else you would like to include or if you have any further questions.
