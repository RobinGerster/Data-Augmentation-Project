from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)
import torch
import torch.nn as nn

class SequenceBertClassifier:
    def __init__(self, device, pretrained_model_name="distilbert-base-uncased", num_labels=2):
        if "distilbert" in pretrained_model_name:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_name)
            self.uses_attention = True
            self.model = DistilBertForSequenceClassification.from_pretrained(
                pretrained_model_name, num_labels=num_labels
            ).to(device)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                pretrained_model_name, num_labels=num_labels
            ).to(device)

    def __call__(self, inputs, attention_mask):
        return self.model(inputs, attention_mask)


class SequenceLSTMClassifier(nn.Module):  # Add inheritance from nn.Module
    def __init__(self, device, pretrained_model_name="bert-base-uncased", num_labels=2):
        super().__init__()  # Initialize the nn.Module
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

        # Define LSTM parameters
        vocab_size = self.tokenizer.vocab_size  # length of the tokenizer's vocab attribute
        embedding_dim = 256
        hidden_dim = 128

        # Create layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim).to(device)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True).to(device)
        self.fc = torch.nn.Linear(hidden_dim, num_labels).to(device)

    def forward(self, inputs):  # Changed __call__ to forward
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
