import pandas as pd

# Replace the file paths with your local paths
train_df = pd.read_json("../Datasets/multinli_1.0/multinli_1.0/multinli_1.0_train.jsonl", lines=True)
val_df = pd.read_json("../Datasets/multinli_1.0/multinli_1.0/multinli_1.0_dev_matched.jsonl", lines=True)
test_df = pd.read_json("../Datasets/multinli_1.0/multinli_1.0/multinli_1.0_dev_mismatched.jsonl", lines=True)


def count_data_splits():
    train_genre_df = train_df['genre']
    val_genre_df = val_df['genre']
    test_genre_df = test_df['genre']

    dataset = [train_genre_df, val_genre_df, test_genre_df]
    splits = ["train", "test", "val"]

    # Get number of examples for each genre
    genre_names = ['fiction', 'government', 'slate', 'telephone', 'travel', 'facetoface', 'letters', 'nineeleven', 'oup', 'verbatim']

    # Count the number of genre examples fpt train, val and test split
    splits_genre_counts = {}
    for i, data in enumerate(dataset):
        genre_counts = {g: 0 for g in genre_names}
        for g in data:
            genre_counts[g] += 1
        splits_genre_counts[splits[i]] = genre_counts


train, val, test = {}, {}, {}

# Define label mapping
label_mapping = {
    'contradiction': 0,
    'entailment': 1,
    'neutral': 2
}

# In-domain genres
genre_matched = ['fiction', 'government', 'slate', 'telephone', 'travel']
# Out-of-domain genres
genre_mismatched = ['facetoface', 'letters', 'nineeleven', 'oup', 'verbatim']


# For each genre, separate the sentences and labels
def group_genres(df, genre, data_dict):
    data_dict[genre]['text'] = [row['sentence1'] + " " + row['sentence2'] for _, row in df.loc[df['genre'] == genre].iterrows() if row['gold_label'] != '-']
    data_dict[genre]['label'] = [label_mapping[row['gold_label']] for _, row in df.loc[df['genre'] == genre].iterrows() if row['gold_label'] != '-']


def process_mnli(split, genre, split_df, new_df, num_samples):
    split[genre] = {}
    group_genres(split_df, genre, split)
    split[genre]['text'] = split[genre]['text'][:num_samples]
    split[genre]['label'] = split[genre]['label'][:num_samples]
    new_df_1 = pd.DataFrame(split[genre], columns=['text', 'label'], index=None)
    return pd.concat([new_df, new_df_1])


# Combine text with label per genre and get the corresponding number of examples for each split
new_train_df = pd.DataFrame()
new_val_df = pd.DataFrame()
for genre in genre_matched:
    new_train_df = process_mnli(train, genre, train_df, new_train_df, 100)
    new_val_df = process_mnli(val, genre, val_df, new_val_df, 200)


new_test_df = pd.DataFrame()
for genre in genre_mismatched:
    new_test_df = process_mnli(test, genre, test_df, new_test_df, 200)

# Save DataFrame to CSV
new_train_df.to_csv('../Datasets/MNLI_ssmba_train.csv', index=False, header=False)
new_val_df.to_csv('../Datasets/MNLI_ssmba_val.csv', index=False, header=False)
new_test_df.to_csv('../Datasets/MNLI_ssmba_test.csv', index=False, header=False)
