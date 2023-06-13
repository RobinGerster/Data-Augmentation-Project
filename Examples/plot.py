import pickle
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

# Load the results
with open('../IMDB_RESULTS_SSMBA.pkl', 'rb') as f:
    results = pickle.load(f)

dataset = "IMDB"
approach = "SSMBA"
conditions = ["bias", "no_bias"]
models = ["BERT", "LSTM", "RNN"]
naugs = [1, 2, 4, 8, 16, 32]
types = ["id"]  #Also use ood but put plots in appendix I think

# Loop over conditions
for i, cond in enumerate(conditions):
    for j, typ in enumerate(types):
        plt.figure(figsize=(10, 6))  # create a new figure for each graph
        for model in models:
            mean_accuracy = [np.mean(results[model][cond][naug][typ]) for naug in naugs]
            plt.plot(naugs, mean_accuracy, label=model)

        plt.xlabel('Number of Augments')
        plt.ylabel('Mean Accuracy')
        # Adjust title depending on the condition
        condition_name = "Naive" if cond == "no_bias" else "Biased"
        plt.title(f'{dataset}: {approach} and {condition_name} {typ} 32 Token Window Performance')
        plt.legend()

        plt.show()
