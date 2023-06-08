import pandas as pd

# Replace the file paths with your local paths
train_df = pd.read_json("../multinli_1.0/multinli_1.0/multinli_1.0_train.jsonl", lines=True)
train_df = train_df.head(40000)
# Define label mapping
label_mapping = {
    "contradiction": 0,
    "entailment": 1,
    "neutral": 2
}

# Combine premise and hypothesis
train_df['text'] = train_df['sentence1'] + " " + train_df['sentence2']

# Map labels to integers
train_df['label'] = train_df['gold_label'].map(label_mapping)

# Select only the 'text' and 'label' columns
new_df = train_df[['text', 'label']]

# Save DataFrame to CSV
new_df.to_csv('mnli_train.csv', index=False)


# Create a new column with the word count of each sentence
train_df['sentence1_length'] = train_df['sentence1'].apply(lambda x: len(x.split()))
train_df['sentence2_length'] = train_df['sentence2'].apply(lambda x: len(x.split()))

# Calculate the average sentence length
avg_sentence1_length = train_df['sentence1_length'].mean()
avg_sentence2_length = train_df['sentence2_length'].mean()

print(f'Average length of sentence1: {avg_sentence1_length}')
print(f'Average length of sentence2: {avg_sentence2_length}')
