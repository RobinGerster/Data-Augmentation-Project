import pandas as pd

# Replace the file paths with your local paths
train_df = pd.read_json("../Datasets/multinli_1.0/multinli_1.0/multinli_1.0_train.jsonl", lines=True)
val_df = pd.read_json("../Datasets/multinli_1.0/multinli_1.0/multinli_1.0_dev_matched.jsonl", lines=True)
test_df = pd.read_json("../Datasets/multinli_1.0/multinli_1.0/multinli_1.0_dev_mismatched.jsonl", lines=True)

train_genre_df = train_df['genre']
val_genre_df = val_df['genre']
test_genre_df = test_df['genre']

# Define label mapping
label_mapping = {
    'contradiction': 0,
    'entailment': 1,
    'neutral': 2
}

dataset = [train_genre_df, val_genre_df, test_genre_df]
splits = ["train", "test", "val"]

# Get number of examples for each genre
genre_names = ['fiction', 'government', 'slate', 'telephone', 'travel', 'facetoface', 'letters', 'nineeleven', 'oup', 'verbatim']

# In-domain genres
genre_matched = ['fiction', 'government', 'slate', 'telephone', 'travel']
# Out-of-domain genres
genre_mismatched = ['facetoface', 'letters', 'nineeleven', 'oup', 'verbatim']


# Count the number of genre examples fpt train, val and test split
splits_genre_counts = {}
for i, data in enumerate(dataset):
    genre_counts = {genre: 0 for genre in genre_names}
    for genre in data:
        genre_counts[genre] += 1
    splits_genre_counts[splits[i]] = genre_counts


# Avoid biasing the data
train, val, test = {}, {}, {}

for genre in genre_matched:
    train[genre] = {}
    val[genre] = {}

for genre in genre_mismatched:
    test[genre] = {}


# For each genre, separate the sentences and labels
def group_genres(df, genre, data_dict):
    data_dict[genre]['text'] = [row['sentence1'] + " " + row['sentence2'] for _, row in df.loc[df['genre'] == genre].iterrows() if row['gold_label'] != '-']
    data_dict[genre]['label'] = [label_mapping[row['gold_label']] for _, row in df.loc[df['genre'] == genre].iterrows() if row['gold_label'] != '-']


# Combine premise and hypothesis
# Map labels to integers
for genre in genre_matched:
    group_genres(train_df, genre, train)
    group_genres(val_df, genre, val)

for genre in genre_mismatched:
    group_genres(test_df, genre, test)

# TODO: combine text with label per genre and get the corresponding number of examples for each split

# # Select only the 'text' and 'label' columns
# new_df = train_df[['text', 'label']]

# # Save DataFrame to CSV
# new_df.to_csv('mnli_train.csv', index=False)

# for genre in genre_matched:
#     print(len(train[genre]['text']))
#     print(len(train[genre]['label']))
#
# print()
# for genre in genre_matched:
#     print(len(val[genre]['text']))
#     print(len(val[genre]['label']))
#
# print()
# for genre in genre_mismatched:
#     print(len(test[genre]['text']))
#     print(len(test[genre]['label']))
