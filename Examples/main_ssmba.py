from Utils.data_loader import get_imdb_ssmba_dataloaders
from Utils.classifiers import SequenceBertClassifier
from Utils.trainer import SupervisedTrainer
import torch
import time, datetime

start_time = time.time()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."

# Get our model
bert = SequenceBertClassifier(device, pretrained_model_name="bert-base-uncased", num_labels=2)

# Get the train dataloader
train_dataloader, imdb_test_dataloader, sst_test_dataloader = get_imdb_ssmba_dataloaders(batch_sizes=[2, 2, 2])

# Initialize additional training parameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(bert.model.parameters(), lr=3e-5)


# Training

supervised_trainer = SupervisedTrainer(bert, criterion, optimizer, train_dataloader, epochs=5, device=device, val_dataloader=imdb_test_dataloader)
supervised_trainer.train()

print(f'Current time: {datetime.datetime.now().time()}, Execution time:',
      time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

# Out-of-Domain Evaluation

start_time = time.time()

ood_performance = supervised_trainer.evaluate(sst_test_dataloader)
print("ood_performance: " + str(ood_performance))

print(f'Current time: {datetime.datetime.now().time()}, Execution time:',
      time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
