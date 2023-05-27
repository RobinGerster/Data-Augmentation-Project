from data_loader import get_dataloader
from models import SequenceBertModel
from trainer import SupervisedTrainer, UDATrainer
import torch

# Get our model
bert = SequenceBertModel(pretrained_model_name="distilbert-base-uncased", num_labels=2)

# Gets our dataloaders
train_dataloader, validation_dataloader, unsup_dataloader = get_dataloader('Datasets/IMDB_500.csv',
                                                                          splits=[0.2, 0.2, 0.6],
                                                                          batch_sizes=[8, 16, 16])

# Initialize additional training parameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(bert.model.parameters(), lr=3e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."


trainer = UDATrainer(bert, optimizer, train_dataloader,unsup_dataloader,epochs=5, device=device, val_dataloader=validation_dataloader)
trainer.train()



