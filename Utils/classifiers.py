from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)
import torch
import torch.nn as nn


class SequenceClassifier:
    def __init__(self, device, pretrained_model_name, num_labels=2):
        self.device = device
        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels
        self.uses_attention = True

    def tokenize(self, text):
        raise NotImplementedError


class SequenceBertClassifier(SequenceClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "distilbert" in self.pretrained_model_name:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.pretrained_model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.pretrained_model_name, num_labels=self.num_labels
            ).to(self.device)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                self.pretrained_model_name, num_labels=self.num_labels
            ).to(self.device)

    def __call__(self, inputs, attention_mask):
        return self.model(inputs, attention_mask).logits


class SequenceLSTMClassifier(SequenceClassifier, nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        super().__init__(*args, **kwargs)
        self.uses_attention = False
        self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_model_name)

        # Define LSTM parameters
        vocab_size = self.tokenizer.vocab_size  # length of the tokenizer's vocab attribute
        embedding_dim = 256
        hidden_dim = 128

        # Create layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim).to(self.device)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True).to(self.device)
        self.fc = torch.nn.Linear(hidden_dim, self.num_labels).to(self.device)

    def forward(self, inputs):
        # Move inputs to device
        inputs = inputs.to(self.device)

        # Pass through embedding layer
        embedded = self.embedding(inputs)

        # Pass through LSTM
        lstm_output, _ = self.lstm(embedded)

        # Take the final LSTM output (ignoring the outputs at each step)
        final_output = lstm_output[:, -1, :]

        # Pass through the final fully connected layer
        logits = self.fc(final_output)

        return logits
