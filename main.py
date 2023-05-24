from data_loader import get_dataloader
from models import SequenceBertModel
from trainer import SupervisedTrainer
import torch

# Get our model
bert = SequenceBertModel(pretrained_model_name="bert-base-uncased")

# Gets our dataloaders
train_dataloader, validation_dataloader, test_dataloader = get_dataloader('Datasets/IMDB_500.csv',
                                                                          splits=[0.7, 0.3, 0],
                                                                          batch_sizes=[32, 32, 32])



# Initialize additional training parameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(bert.model.parameters(), lr=3e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."

supervised_trainer = SupervisedTrainer(bert, criterion, optimizer, train_dataloader, device=device,
                                       val_dataloader=validation_dataloader)
supervised_trainer.train()