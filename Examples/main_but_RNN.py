from Utils.data_loader import get_dataloader
from Utils.classifiers import SequenceRNNClassifier
from Utils.trainer import RNNTrainer, SupervisedTrainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."

# Get our model
rnn = SequenceRNNClassifier(device, pretrained_model_name="distilbert-base-uncased", num_labels=2,max_length=512)

# Gets our dataloaders
train_dataloader, validation_dataloader, test_dataloader = get_dataloader('../Datasets/IMDB_Full.csv',
                                                                          splits=[0.5, 0.5, 0.0],
                                                                          batch_sizes=[16, 32, 32])

# Initialize additional training parameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(rnn.parameters(), lr=3e-3)

assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."


supervised_trainer = SupervisedTrainer(rnn, criterion, optimizer, train_dataloader, epochs=25, device=device, val_dataloader=validation_dataloader)
supervised_trainer.train()