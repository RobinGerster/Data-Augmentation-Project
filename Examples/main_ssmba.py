from Utils.data_loader import get_dataloader, get_imdb_training_test_data
from Utils.classifiers import SequenceBertClassifier
from Utils.trainer import SupervisedTrainer
import torch
import pandas as pd
import time

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."

# Get our model
bert = SequenceBertClassifier(device, pretrained_model_name="bert-base-uncased", num_labels=2)

# Get the train dataloader
# train_dataloader = get_dataloader('../Datasets/IMDB_40k_ssmba_train.csv', splits=[1.0, 0.0, 0.0], batch_sizes=[2, 2, 2])
train_dataloader, test_dataloader = get_imdb_training_test_data(batch_sizes=[2, 2])

# # Initialize additional training parameters
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(bert.model.parameters(), lr=3e-5)
#
#
# # Training
#
# supervised_trainer = SupervisedTrainer(bert, criterion, optimizer, train_dataloader, epochs=5, device=device)
# supervised_trainer.train()
#
# print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
#
# # Evaluation
#
# start_time = time.time()
#
# supervised_trainer.evaluate(test_dataloader)
#
# print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
