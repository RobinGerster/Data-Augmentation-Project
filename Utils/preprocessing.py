import csv
import os
from datasets import load_dataset
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')


def preprocess_imdb_for_ssmba_augmentation(dataset, bias):
    if bias:
        save_folder = 'bias'
    else:
        save_folder = 'no_bias'

    # Read the training data
    if dataset == "IMDB":
        if bias:
            train_set = '../Datasets/IMDB_500_bias.csv'
        else:
            train_set = '../Datasets/IMDB_500.csv'
        path = r'../Datasets/ssmba/mnli_' + save_folder

    if dataset == "MNLI":
        if bias:
            train_set = '../Datasets/MNLI_ssmba_bias_train.csv'
        else:
            train_set = '../Datasets/MNLI_ssmba_train.csv'
        path = r'../Datasets/ssmba/mnli_' + save_folder

    # Create folder to save data prepared for augmentation
    os.makedirs(path, exist_ok=True)

    # Prepare the data for augmentation
    with open(train_set, "r", encoding="utf8") as csvfile, \
            open(os.path.join(path, "input.txt"), "w", encoding="utf8") as input_file, \
            open(os.path.join(path, "labels.txt"), "w") as labels_file:
        csvReader = csv.reader(csvfile, delimiter=',')
        for row in csvReader:
            labels_file.write(row[0] + '\n')
            input_file.write(row[1] + '\n')


def ssmba_augmented_to_csv(dataset, bias):
    for naug in [1, 2, 4, 8, 16, 32]:
        if bias:
            sentiment = "bias"
        else:
            sentiment = "no_bias"

        # Read the original data
        if dataset == "IMDB":
            if not bias:
                original_df = pd.read_csv("../Datasets/IMDB_500.csv", header=None, names=["label", "text"])
            else:
                original_df = pd.read_csv("../Datasets/IMDB_500_bias.csv", header=None, names=["label", "text"])
            sentences = "ssmba_out_imdb_500_" + sentiment + "_" + str(naug)
            labels = "ssmba_out_imdb_500_" + sentiment + "_" + str(naug) + ".label"
            output = "../Datasets/imdb_" + sentiment + "/IMDB_500_" + sentiment + "_" + str(naug) + "_ssmba_train.csv"
            path = "../Datasets/mnli_" + sentiment

        if dataset == "MNLI":
            if not bias:
                original_df = pd.read_csv("../Datasets/MNLI_ssmba_train.csv", header=None, names=["label", "text"])
            else:
                original_df = pd.read_csv("../Datasets/MNLI_ssmba_bias_train.csv", header=None, names=["label", "text"])
            sentences = "ssmba_out_mnli_" + sentiment + "_" + str(naug)
            labels = "ssmba_out_mnli_" + sentiment + "_" + str(naug) + ".label"
            output = "../Datasets/mnli_" + sentiment + "/MNLI_" + sentiment + "_" + str(naug) + "_ssmba_train.csv"
            path = "../Datasets/mnli_" + sentiment

        # Create folder to save the augmented data
        os.makedirs(path, exist_ok=True)

        original_text = original_df["text"]
        original_labels = original_df["label"]

        # Prepare the augmented data for training and combine the original with the augmented data
        with open(os.path.join(path, sentences), "r", encoding="utf8") as input_augmented, \
                open(os.path.join(path, labels), "r") as labels_augmented, \
                open(output, "w", encoding="utf8") as csvfile:
            reader1 = labels_augmented.readlines()
            reader2 = input_augmented.readlines()
            writer = csv.writer(csvfile, lineterminator='\n')

            stop_words = set(stopwords.words('english'))

            # Combine augmented text with labels
            for line1, line2 in zip(reader1, reader2):
                value1 = int(line1.strip())
                value2 = line2.rstrip()
                print(value2)
                # Remove stop words
                word_tokens = word_tokenize(value2)
                filtered_sentence = [w for w in word_tokens if w not in stop_words]
                value2 = " ".join(filtered_sentence)
                print(value2)
                writer.writerow([value1, value2])
                break

            # Add original data
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
    # preprocess_imdb_for_ssmba_augmentation(dataset="MNLI", bias=True)
    # ssmba_augmented_to_csv(bias=True)
