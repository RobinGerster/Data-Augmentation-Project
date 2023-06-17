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
types = ["id"]  # Also use ood but put plots in appendix

plt.figure(figsize=(10, 6))  # create a single figure for both conditions

# Define colors for each model
colors = ['blue', 'red', 'green']  # Add more colors if needed

# Loop over models
for i, model in enumerate(models):
    # Get the color for the model
    color = colors[i % len(colors)]

    # Loop over conditions
    for j, cond in enumerate(conditions):
        for k, typ in enumerate(types):
            mean_accuracy = [np.mean(results[model][cond][naug][typ]) for naug in naugs]

            # Determine the marker type based on the condition
            marker = 'o' if cond == "no_bias" else 'x'

            # Plot data with lines and markers, and label
            c = "Biased"
            if cond == "no_bias":
                c = "Naive"
            plt.plot(naugs, mean_accuracy, marker=marker, color=color, label=f'{model} ({c})', markersize=10)

plt.xlabel('Number of Augments', fontsize=18)
plt.ylabel('Mean Accuracy', fontsize=18)
plt.title(f'{dataset}: {approach} 32 Token Window Performance', fontsize=18)
plt.legend()
plt.show()


