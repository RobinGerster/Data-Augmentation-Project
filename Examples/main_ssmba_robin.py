from Utils.data_loader import get_imdb_ssmba_dataloaders
from Utils.classifiers import SequenceBertClassifier, SequenceLSTMClassifier
from Utils.trainer import SupervisedTrainer
import torch
import time
from tqdm import tqdm
import pickle

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."

ood_performance_list = []
id_performance_list = []

dataset = "IMDB" #Maybe do it for one dataset at a time and save results run again for "MNLI"
models = ["BERT", "LSTM", "RNN"]
conditions = ["no_bias", "bias"]
naug = [1, 2, 4, 8, 16, 32]
runs = 5
results = {}

total_runs = len(dataset) * len(conditions) * len(naug) * runs
pbar = tqdm(total=total_runs, desc='Progress...')

for model in models:
    for condition in conditions:
        for n in naug:
            id_performance_list = []
            ood_performance_list = []
            for run in runs:
                # Get our model
                if dataset == "IMDB":
                    labels = 2
                else:
                    labels = 3
                if model == "BERT":
                    m = SequenceBertClassifier(device, pretrained_model_name="bert-base-uncased", num_labels=labels)
                elif model == "LSTM":
                    m = SequenceLSTMClassifier(device, pretrained_model_name="bert-base-uncased", num_labels=labels)
                elif model == "RNN":
                    raise NotImplementedError("RNN not yet implemented")

                # Get the train dataloader
                train_dataloader, imdb_test_dataloader, sst_test_dataloader = get_ssmba_dataloaders(
                    batch_sizes=[16, 32, 32],
                    naug=naug, bias = condition == "bias", dataset=dataset)

                # Initialize additional training parameters
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.AdamW(m.model.parameters(), lr=3e-5)

                # Training
                supervised_trainer = SupervisedTrainer(m, criterion, optimizer, train_dataloader,
                                                       id_performance_list,
                                                       epochs=5,
                                                       device=device,
                                                       val_dataloader=imdb_test_dataloader)
                supervised_trainer.train()
                id_performance_list.append(
                    max(supervised_trainer.id_performance_list))  # only care about max acc for each run. Assuming supervised_trainer stores accuracies at every epoch

                # Out-of-Domain Evaluation
                ood_performance = supervised_trainer.evaluate(sst_test_dataloader)
                ood_performance_list.append(max(ood_performance))

                pbar.update()  # increment the progress bar
            results[model][condition][naug]['id'] = id_performance_list #Should be a list on runs many max accuracies
            results[model][condition][naug]['ood'] = ood_performance_list  #Should be a list on runs many max accuracies

# We have everything we need for IMDB so we save the data. Later we can read it again for plotting, tables, rank test etc

# Write to file
with open(f'{dataset}_RESULTS_SSMBA.pkl', 'wb') as f:
    pickle.dump(results, f)