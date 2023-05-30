from data_loader import get_dataloader
from classifiers import SequenceBertClassifier, SequenceLSTMClassifier
from trainer import SupervisedTrainer, UDATrainer, LSTMTrainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."

# Get our model
lstm = SequenceLSTMClassifier(device, pretrained_model_name="distilbert-base-uncased", num_labels=2)

# Gets our dataloaders
train_dataloader, validation_dataloader, test_dataloader = get_dataloader('Datasets/IMDB_Full.csv',
                                                                          splits=[0.5, 0.5, 0.0],
                                                                          batch_sizes=[128, 128, 128])

# Initialize additional training parameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(lstm.parameters(), lr=3e-3)

assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."


supervised_trainer = LSTMTrainer(lstm, criterion, optimizer, train_dataloader, epochs=50, device=device, val_dataloader=validation_dataloader)
supervised_trainer.train()



