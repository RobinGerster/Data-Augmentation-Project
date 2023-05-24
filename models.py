from transformers import BertForSequenceClassification, BertTokenizerFast

class SequenceBertModel:
    def __init__(self, pretrained_model_name="distilbert-base-uncased"):
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
        self.model = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)

    def __call__(self, inputs):
        return self.model(**inputs)
