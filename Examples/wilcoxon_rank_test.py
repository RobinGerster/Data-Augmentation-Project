import pickle
import numpy as np
from scipy.stats import wilcoxon

#Expecting results in format results[model][condition][naug]['id' or 'ood'] !!!!!!!!!!!!!!!

#Load your results
with open('path to pkl', 'rb') as f:
    results = pickle.load(f)

model = "BERT"
for type in ["id", "ood"]:
    # We first want to print out the table information we need for the report
    for condition in results[model]:
        print(f"============== {condition} : {type} =============")
        for naug in results[model][condition]:
            mean_acc = np.mean(results[model][condition][naug][type])
            std = np.std(results[model][condition][naug][type])
            print(f"Naug:{naug} Mean Accuracy:{round(float(mean_acc),4)}, STD:{round(float(std),4)}")

#Now we actually want to report statistical significance with the rank test

for type in ["id", "ood"]:
    # First, gather all the accuracies per condition
    condition_accuracies = {}
    for condition in results[model]:
        for naug in results[model][condition]:
            if condition not in condition_accuracies:
                condition_accuracies[condition] = []
            condition_accuracies[condition].extend(results[model][condition][naug][type])

    # Then, do the pairwise Wilcoxon tests
    w, p = wilcoxon(condition_accuracies["no_bias"], condition_accuracies["bias"])
    print(f"Wilcoxon test for type: {type} gives W = {w}, p = {p}")




