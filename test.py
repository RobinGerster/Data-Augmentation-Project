# Install the required transformers library.
# !pip install transformers

import torch
from transformers import MarianMTModel, MarianTokenizer


class BackTranslation:
    def __init__(self):
        # Create source language tokenizer and model
        self.tokenizer_src = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
        self.model_src = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')

        # Create target language tokenizer and model
        self.tokenizer_tgt = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
        self.model_tgt = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')

    def translate(self, text, model, tokenizer):
        """Translate a given text using the provided model and tokenizer."""
        encoded_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
        translation_logits = model.generate(**encoded_text)
        decoded_translation = tokenizer.batch_decode(translation_logits, skip_special_tokens=True)[0]

        return decoded_translation

    def backtranslate(self, text, n=1):
        """Backtranslate a given text n times."""
        augmented_text = text

        for i in range(n):
            # Translate to French
            fr_text = self.translate(augmented_text, self.model_src, self.tokenizer_src)

            # Translate back to English
            en_text = self.translate(fr_text, self.model_tgt, self.tokenizer_tgt)

            # Use this augmented text as the basis for the next augmentation
            augmented_text = en_text

            print(f"Augmentation {i + 1}: {augmented_text}")


# Test the script
bt = BackTranslation()
sentence = "This day is very nice but I wish I could have some ice cream as well."
augment_num = 10

bt.backtranslate(sentence, augment_num)
