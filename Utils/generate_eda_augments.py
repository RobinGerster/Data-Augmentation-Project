from EDA.augment import gen_eda
import csv
import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

SOURCE_PATH="../Datasets/MNLI_ssmba_train.csv"
DESTINATION_FOLDER="../Datasets/eda_data/MNLI"
naugs=[1, 2, 4, 8, 16, 32]
# Synonyms
sr=0.4
# Random delete
rd=0.05
# Random insertion
ri=0
#Random swap
rs=0.01
def eda_augments(SOURCE_PATH,DESTINATION_FOLDER,naugs):
    for naug in naugs:
        gen_eda(SOURCE_PATH, f"{DESTINATION_FOLDER}/MNLI_eda{naug}.csv", alpha_sr=sr,
                alpha_rd=rd, alpha_ri=ri, alpha_rs=rs, num_aug=naug)




def bias_mnli_data(split, naug=0):
    if split == "train":
        data_path = f'../Datasets/eda_data/MNLI/MNLI_eda{naug}.csv'
        #os.makedirs('../Datasets/mnli_bias', exist_ok=True)
        save_path = f'../Datasets/eda_data/MNLI/MNLI_eda{naug}_bias.csv'
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

def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    sentence = " ".join(filtered_sentence)
    return sentence


if __name__ == "__main__":
    # eda_augments(SOURCE_PATH,DESTINATION_FOLDER,naugs)
    for naug in naugs:
        bias_mnli_data("train",naug=naug)

    bias_mnli_data('test')
    bias_mnli_data('validation')