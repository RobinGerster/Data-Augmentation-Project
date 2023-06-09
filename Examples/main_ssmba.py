from Utils.data_loader import get_dataloader
from Utils.classifiers import SequenceBertClassifier
from Utils.trainer import SupervisedTrainer
import torch
import time

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."

# Get our model
bert = SequenceBertClassifier(device, pretrained_model_name="distilbert-base-uncased", num_labels=2)

# Gets our dataloaders
train_dataloader, validation_dataloader, test_dataloader = get_dataloader('../Datasets/IMDB_500_ssmba_augmented_8.csv',
                                                                          splits=[0.2, 0.2, 0.6],
                                                                          batch_sizes=[16, 32, 32])

# Initialize additional training parameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(bert.model.parameters(), lr=3e-5)

# assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."


supervised_trainer = SupervisedTrainer(bert, criterion, optimizer, train_dataloader, epochs=5, device=device, val_dataloader=validation_dataloader)
supervised_trainer.train()

print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
