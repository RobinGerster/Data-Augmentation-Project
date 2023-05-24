from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

class SequenceBertModel:
    def __init__(self, pretrained_model_name="distilbert-base-uncased", num_labels=2):
        if "distilbert" in pretrained_model_name:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                pretrained_model_name, num_labels=num_labels
            )
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                pretrained_model_name, num_labels=num_labels
            )

    def __call__(self, inputs):
        return self.model(**inputs)
