from Utils.data_loader import get_imdb_ssmba_dataloaders
from Utils.classifiers import SequenceBertClassifier
from Utils.trainer import SupervisedTrainer
import torch
import time

start_time = time.time()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."

# Get our model
bert = SequenceBertClassifier(device, pretrained_model_name="bert-base-uncased", num_labels=2)

# Get the train dataloader
train_dataloader, imdb_test_dataloader, sst_test_dataloader = get_imdb_ssmba_dataloaders(batch_sizes=[2, 2, 2])
print(len(train_dataloader))
print(len(imdb_test_dataloader))
print(len(sst_test_dataloader))

# Initialize additional training parameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(bert.model.parameters(), lr=3e-5)


# Training

supervised_trainer = SupervisedTrainer(bert, criterion, optimizer, train_dataloader, epochs=5, device=device)
supervised_trainer.train()

print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

# In-Domain Evaluation

start_time = time.time()

supervised_trainer.evaluate(imdb_test_dataloader)

print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

# Out-of-Domain Evaluation

start_time = time.time()

supervised_trainer.evaluate(sst_test_dataloader)

print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
