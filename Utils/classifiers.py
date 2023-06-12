from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Base class for a sequence classifier
class SequenceClassifier:
    def __init__(self, device, pretrained_model_name, num_labels=2):
        self.device = device  # The PyTorch device (CPU or GPU) to use for model computations
        self.pretrained_model_name = pretrained_model_name  # The name of the pre-trained model to use
        self.num_labels = num_labels  # The number of output labels
        self.uses_attention = True  # Whether this model uses attention

    def tokenize(self, text):
        raise NotImplementedError  # This should be implemented in each subclass


# Sequence classifier using a BERT model
class SequenceBertClassifier(SequenceClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if the specified pre-trained model is a DistilBERT model
        if "distilbert" in self.pretrained_model_name:
            # If it is, use the corresponding tokenizer and model
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.pretrained_model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.pretrained_model_name, num_labels=self.num_labels
            ).to(self.device)
        else:
            # Otherwise, use the BERT tokenizer and model
            self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                self.pretrained_model_name, num_labels=self.num_labels
            ).to(self.device)

    def __call__(self, inputs, attention_mask):
        # Call the model with the inputs and attention mask, and return the logits
        return self.model(inputs, attention_mask).logits


# Sequence classifier using an LSTM model
class SequenceLSTMClassifier(SequenceClassifier, nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        super().__init__(*args, **kwargs)
        self.uses_attention = False  # This model doesn't use attention
        self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_model_name)  # Use the BERT tokenizer

        # Define LSTM parameters
        vocab_size = self.tokenizer.vocab_size  # Length of the tokenizer's vocab attribute
        embedding_dim = 256  # Dimension of the embedding layer
        hidden_dim = 128  # Dimension of the hidden state in the LSTM

        # Create layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim).to(self.device)  # Embedding layer
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True).to(self.device)  # LSTM layer
        self.fc = torch.nn.Linear(hidden_dim, self.num_labels).to(self.device)  # Fully connected output layer

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


class SequenceRNNClassifier(SequenceClassifier, nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        super().__init__(*args, **kwargs)
        #self.uses_attention = False  # This model doesn't use attention
        if "distilbert" in self.pretrained_model_name:
            # If it is, use the corresponding tokenizer and model
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.pretrained_model_name)
        else:
            # Otherwise, use the BERT tokenizer and model
            self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_model_name)
        # Define LSTM parameters
        vocab_size = self.tokenizer.vocab_size  # Length of the tokenizer's vocab attribute
        embedding_dim = 256  # Dimension of the embedding layer
        hidden_dim = 128  # Dimension of the hidden state in the LSTM

        # Create layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim).to(self.device)  # Embedding layer
        self.rnn=nn.GRU(embedding_dim,hidden_dim,bidirectional=True,batch_first=True).to(
            self.device)  # LSTM layer
        self.bn=nn.BatchNorm1d(hidden_dim*2).to(self.device)
        self.fc = torch.nn.Linear(hidden_dim*2, self.num_labels).to(self.device)  # Fully connected output layer

    def forward(self,input_ids,attention_mask=None):
        input_ids=input_ids.to(self.device)
        attention_mask=attention_mask.to(self.device)
        seq_lengths=attention_mask.sum(dim=1)
        embedded=self.embedding(input_ids)
        row_indices=torch.arange(input_ids.size(0))
        packed=pack_padded_sequence(embedded,attention_mask.sum(dim=1).cpu(),batch_first=True,enforce_sorted=False)
        packed_op,_=self.rnn(packed)
        unpacked_op,_=pad_packed_sequence(packed_op,batch_first=True)
        #print(unpacked_op[:,seq_lengths-1,:].shape)

        latest_output=unpacked_op[row_indices,seq_lengths-1,:]
        #print(latest_output.shape)
        #normed=self.bn(latest_output)
        logits=self.fc(latest_output)
        return logits


class SequenceRNNClassifier(SequenceClassifier, nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        super().__init__(*args, **kwargs)
        #self.uses_attention = False  # This model doesn't use attention
        if "distilbert" in self.pretrained_model_name:
            # If it is, use the corresponding tokenizer and model
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.pretrained_model_name)
        else:
            # Otherwise, use the BERT tokenizer and model
            self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_model_name)
        # Define LSTM parameters
        vocab_size = self.tokenizer.vocab_size  # Length of the tokenizer's vocab attribute
        embedding_dim = 256  # Dimension of the embedding layer
        hidden_dim = 128  # Dimension of the hidden state in the LSTM

        # Create layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim).to(self.device)  # Embedding layer
        self.rnn=nn.GRU(embedding_dim,hidden_dim,bidirectional=True,batch_first=True).to(
            self.device)  # LSTM layer
        self.bn=nn.BatchNorm1d(hidden_dim*2).to(self.device)
        self.fc = torch.nn.Linear(hidden_dim*2, self.num_labels).to(self.device)  # Fully connected output layer

    def forward(self,input_ids,attention_mask=None):
        input_ids=input_ids.to(self.device)
        attention_mask=attention_mask.to(self.device)
        seq_lengths=attention_mask.sum(dim=1)
        embedded=self.embedding(input_ids)
        row_indices=torch.arange(input_ids.size(0))
        packed=pack_padded_sequence(embedded,attention_mask.sum(dim=1).cpu(),batch_first=True,enforce_sorted=False)
        packed_op,_=self.rnn(packed)
        unpacked_op,_=pad_packed_sequence(packed_op,batch_first=True)
        #print(unpacked_op[:,seq_lengths-1,:].shape)

        latest_output=unpacked_op[row_indices,seq_lengths-1,:]
        #print(latest_output.shape)
        #normed=self.bn(latest_output)
        logits=self.fc(latest_output)
        return logits
