import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from afinn import Afinn


nltk.download('stopwords')


def apply_inductive_bias(split, dataset="IMDB"):
    if split == "train":
        for naug in [1, 2, 4, 8, 16, 32]:
            if dataset == "IMDB":
                extract_inductive_bias(split, naug=naug)
            else:
                bias_mnli_data(split, naug=naug)
    elif split == "validation":
        if dataset == "IMDB":
            extract_inductive_bias(split)
        else:
            bias_mnli_data(split)
    elif split == "test":
        if dataset == "IMDB":
            extract_inductive_bias(split)
        else:
            bias_mnli_data(split)


def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    sentence = " ".join(filtered_sentence)
    return sentence


def bias_mnli_data(split, naug=0):
    if split == "train":
        data_path = '../Datasets/mnli_no_bias/MNLI_no_bias_' + str(naug) + '_ssmba_train.csv'
        os.makedirs('../Datasets/mnli_bias', exist_ok=True)
        save_path = '../Datasets/mnli_bias/MNLI_bias_' + str(naug) + '_ssmba_train.csv'
    elif split == "validation":
        data_path = '../Datasets/MNLI_ssmba_val.csv'
        save_path = '../Datasets/MNLI_ssmba_bias_val.csv'
    elif split == "test":
        data_path = '../Datasets/MNLI_ssmba_test.csv'
        save_path = '../Datasets/MNLI_ssmba_bias_test.csv'

    # Load the dataset
    df = pd.read_csv(data_path, header=None, names=['label', 'text'])
    new_df = pd.DataFrame(columns=['label', 'text'])

    for _, row in df.iterrows():
        text = remove_stopwords(row['text'])
        new_row = pd.DataFrame({'text': [text], 'label': [row['label']]})
        new_df = pd.concat([new_df, new_row], ignore_index=True)

    new_df.to_csv(save_path, index=False, header=False)


def extract_inductive_bias(split, naug=0):
    # Initialize Afinn
    afinn = Afinn()

    # DEFINE THE DATASET PATH YOU WANT TO PRUNE
    if split == "train":
        data_path = '../Datasets/imdb_no_bias/IMDB_no_bias_' + str(naug) + '_ssmba_train.csv'
        os.makedirs('../Datasets/imdb_bias', exist_ok=True)
        save_path = '../Datasets/imdb_bias/IMDB_bias_' + str(naug) + '_ssmba_train.csv'
    elif split == "validation":
        data_path = '../Datasets/IMDB_1000_ssmba_val.csv'
        save_path = '../Datasets/IMDB_1000_ssmba_bias_val.csv'
    elif split == "test":
        data_path = '../Datasets/SST-2_1000_ssmba_test.csv'
        save_path = '../Datasets/SST-2_1000_ssmba_bias_test.csv'

    # Load the dataset
    df = pd.read_csv(data_path, header=None, names=['label', 'text'])

    # Tags we are interested in
    interesting_tags = ['JJ', 'JJR', 'JJS']

    # Window size
    window_size = 32

    new_df = pd.DataFrame(columns=['label', 'text'])

    for index, row in df.iterrows():
        sentences = sent_tokenize(row['text'])
        max_valence = -1
        max_interesting_window = ""
        non_first_windows_count = 0
        for i in range(len(sentences)):
            tokens = word_tokenize(sentences[i])
            k = i+1
            while len(tokens) < 32 and k<len(sentences):
                needed_tokens = 32-len(tokens)
                # Get tokens from the k-th sentence
                next_sentence_tokens = word_tokenize(sentences[k])
                # Append the needed tokens from the next sentence to the current tokens
                tokens.extend(next_sentence_tokens[:needed_tokens])
                k += 1

            tokens_tags = pos_tag(tokens)

            # Consider a window of size 32
            if len(tokens_tags) > window_size:
                tokens_tags = tokens_tags[:window_size]

            #Calculate sentiment
            total_sentiment = 0
            for word, tag in tokens_tags:
                if tag in interesting_tags:
                    total_sentiment += afinn.score(word)

            # If the count is more than the max found before, keep track of this window
            if abs(total_sentiment) > max_valence:
                max_valence = abs(total_sentiment)
                max_interesting_window = " ".join([word for word, tag in tokens_tags])
                if i == 0:
                    non_first_windows_count += 1

        #     print(" ".join([word for word, tag in tokens_tags]))
        #     print(f'Sentiment score: {total_sentiment}')
        # print(max_interesting_window)
        # print("Changed " + str(non_first_windows_count) +" of "+ str(len(df)))

        new_row = pd.DataFrame({'text': [max_interesting_window], 'label': [row['label']]})
        new_df = pd.concat([new_df, new_row], ignore_index=True)

    new_df.to_csv(save_path, index=False, header=False)


if __name__ == "__main__":
    # "IMDB",
    for dataset in ["IMDB"]:
        for split in ["train", "validation", "test"]:
            apply_inductive_bias(split=split, dataset=dataset)
