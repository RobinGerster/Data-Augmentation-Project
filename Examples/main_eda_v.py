from Utils.data_loader import get_ssmba_dataloaders,get_dataloader
from Utils.classifiers import SequenceBertClassifier, SequenceLSTMClassifier, SequenceRNNClassifier
from Utils.trainer import SupervisedTrainer
import torch
import time
from tqdm import tqdm
import pickle

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."

dataset = "IMDB"  # Maybe do it for one dataset at a time and save results run again for "MNLI"
models = ["BERT", "LSTM", "RNN"]

conditions = ["no_bias", "bias"]
naug = [1, 2, 4, 8, 16, 32]
runs = 5
epochs = 10
results = {}

total_runs = len(conditions) * len(naug) * runs * len(models)
pbar = tqdm(total=total_runs, desc='Progress...')

for model in models:
    results[model] = {}
    for condition in conditions:
        results[model][condition] = {}
        for n in naug:
            results[model][condition][n] = {}
            id_performance_list = []
            ood_performance_list = []
            for run in range(1, runs + 1):
                # Get our model
                if dataset == "IMDB":
                    labels = 2
                else:
                    labels = 3
                if model == "BERT":
                    m = SequenceBertClassifier(device, pretrained_model_name="distilbert-base-uncased",
                                               num_labels=labels)
                elif model == "LSTM":
                    m = SequenceLSTMClassifier(device, pretrained_model_name="distilbert-base-uncased",
                                               num_labels=labels)
                elif model == "RNN":
                    m = SequenceRNNClassifier(device, pretrained_model_name="distilbert-base-uncased",
                                              num_labels=labels)

                # Get the train dataloader
                # train_dataloader, imdb_test_dataloader, sst_test_dataloader = get_ssmba_dataloaders(
                #     batch_sizes=[16, 32, 32],
                #     naug=n, bias=condition == "bias", dataset=dataset)
                if condition=="bias":
                    train_dataloader=get_dataloader(f"../Datasets/eda_data/IMDB_{(n+1)*500}_eda{n}_sentiment.csv")[0]
                    imdb_test_dataloader = get_dataloader("../Datasets/IMDB_1000_ssmba_bias_val.csv")[0]
                    sst_test_dataloader = get_dataloader("../Datasets/SST-2_1000_ssmba_bias_test.csv")[0]
                else:
                    train_dataloader=get_dataloader(f"../Datasets/eda_data/IMDB_{(n+1)*500}_eda{n}.csv")[0]
                    imdb_test_dataloader=get_dataloader("../Datasets/IMDB_1000_ssmba_val.csv")[0]
                    sst_test_dataloader=get_dataloader("../Datasets/SST-2_1000_ssmba_test.csv")[0]
                # Initialize additional training parameters
                criterion = torch.nn.CrossEntropyLoss()
                if model == "BERT":
                    optimizer = torch.optim.AdamW(m.model.parameters(), lr=3e-5)
                else:
                    optimizer = torch.optim.AdamW(m.parameters(), lr=3e-5)

                # Training
                supervised_trainer = SupervisedTrainer(m, criterion, optimizer, train_dataloader,
                                                       id_performance_list,
                                                       epochs=epochs,
                                                       device=device,
                                                       val_dataloader=imdb_test_dataloader)
                supervised_trainer.train()

                # Out-of-Domain Evaluation
                ood_performance = supervised_trainer.evaluate(sst_test_dataloader)
                ood_performance_list.append(ood_performance)

                pbar.update()  # increment the progress bar

            id_performance_list = [max(id_performance_list[i - runs:i]) for i in
                                   range(epochs, runs * epochs + 1, epochs)]
            results[model][condition][n]['id'] = id_performance_list  # Should be a list on runs many max accuracies
            results[model][condition][n]['ood'] = ood_performance_list  # Should be a list on runs many max accuracies

# We have everything we need for IMDB so we save the data. Later we can read it again for plotting, tables, rank test etc

# Write to file
with open(f'../{dataset}_RESULTS_EDA.pkl', 'wb') as f:
    pickle.dump(results, f)