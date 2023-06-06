import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from collections import deque
from afinn import Afinn

# Initialize Afinn
afinn = Afinn()

#DEFINE THE DATASET PATH YOU WANT TO PRUNE
data_path = 'IMDB_Dataset_updated.csv'
save_path = 'IMDB_Dataset_updated_sentiment.csv'

# Load the dataset
df = pd.read_csv(data_path)

# Tags we are interested in
interesting_tags = ['JJ', 'JJR', 'JJS']

# Window size
window_size = 32

new_df = pd.DataFrame(columns=['text', 'label'])

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

        print(" ".join([word for word, tag in tokens_tags]))
        print(f'Sentiment score: {total_sentiment}')
    print(max_interesting_window)
    print("Changed " + str(non_first_windows_count) +" of "+ str(len(df)))

    new_row = pd.DataFrame({'text': [max_interesting_window], 'label': [row['label']]})
    new_df = pd.concat([new_df, new_row], ignore_index=True)

new_df.to_csv(save_path, index=False)

print(new_df)
