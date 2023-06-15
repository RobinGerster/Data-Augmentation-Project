import csv
import os
from datasets import load_dataset
import pandas as pd


def preprocess_imdb_for_ssmba_augmentation():
    path = r'../Datasets/ssmba/bias'
    os.makedirs(path, exist_ok=True)

    with open('../Datasets/IMDB_500_bias.csv', "r", encoding="utf8") as csvfile, \
            open(os.path.join(path, "input.txt"), "w", encoding="utf8") as input_file, \
            open(os.path.join(path, "labels.txt"), "w") as labels_file:
        csvReader = csv.reader(csvfile, delimiter=',')
        for row in csvReader:
            labels_file.write(row[0] + '\n')
            input_file.write(row[1] + '\n')


def ssmba_augmented_to_csv(bias=False):
    for naug in [1, 2, 4, 8, 16, 32]:
        path = r'../Datasets/ssmba/bias'
        os.makedirs(path, exist_ok=True)

        if not bias:
            original_df = pd.read_csv("../Datasets/IMDB_500.csv", header=None, names=["label", "text"])
            original_text = original_df["text"]
            original_labels = original_df["label"]
        else:
            original_df = pd.read_csv("../Datasets/IMDB_500_bias.csv", header=None, names=["label", "text"])
            original_text = original_df["text"]
            original_labels = original_df["label"]

        if bias:
            sentiment = "bias_"
        else:
            sentiment = ""

        with open(os.path.join(path, "ssmba_out_imdb_500_" + sentiment + str(naug)), "r", encoding="utf8") as input_augmented, \
                open(os.path.join(path, "ssmba_out_imdb_500_" + sentiment + str(naug) + ".label"), "r") as labels_augmented, \
                open('../Datasets/IMDB_500_' + sentiment + str(naug) + '_ssmba_train.csv', "w", encoding="utf8") as csvfile:
            reader1 = labels_augmented.readlines()
            reader2 = input_augmented.readlines()
            writer = csv.writer(csvfile, lineterminator='\n')

            # Iterate through each line and write the sum to the CSV file
            for line1, line2 in zip(reader1, reader2):
                value1 = int(line1.strip())
                value2 = line2.rstrip()
                writer.writerow([value1, value2])

            for label, text in zip(original_labels, original_text):
                writer.writerow([label, text])


def prepare_sst2_test():
    dataset = load_dataset('glue', 'sst2', split='train')
    df = pd.DataFrame(columns=['label', 'text'])
    df['text'] = dataset["sentence"]
    df['label'] = dataset["label"]

    new_df = pd.DataFrame(columns=['label', 'text'])
    for i, sentiment in enumerate(df['text']):
        if len(sentiment.split()) > 32:
            new_df.loc[i] = df.loc[i]
        if len(new_df) == 1000:
            break

    new_df.to_csv('../Datasets/SST-2_1000_ssmba_test.csv', index=False, header=False)


def prepare_imdb_val(save=False):
    # Specify there is no header in the file
    df = pd.read_csv("../Datasets/IMDB_Full.csv", header=None)
    df1 = df.head(500)
    df2 = df.head(1500)

    # Contain 100 examples
    test_df = df2.drop(df1.index)

    # Save these examples in a separate file for ssmba augmentation
    if save:
        test_df.to_csv('../Datasets/IMDB_1000_ssmba_val.csv', index=False, header=False)


# if __name__ == "__main__":
    # preprocess_imdb_for_ssmba_augmentation()
    # ssmba_augmented_to_csv(bias=True)
