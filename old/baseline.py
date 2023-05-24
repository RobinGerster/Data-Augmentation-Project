import torch
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader
import numpy as np
from data_loader import get_dataloader

accuracies = []
for i in range(10):
    # Constants
    model_name = "bert-base-uncased"
    num_labels = 2
    batch_size = 8
    num_epochs = 10
    lr = 3e-5
    uda_coeff = 1.0  # The coefficient for UDA loss

    # Setting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset = load_dataset("imdb")

    # Shuffle the datasets
    train_dataset = dataset['train'].shuffle().select(range(500))
    val_dataset = dataset['test'].shuffle().select(range(55))  # select more for splitting
    test_dataset = dataset['test'].shuffle().select(range(55))  # select more for splitting

    # Initialize tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model = model.to(device)



    # Define data loaders
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size)
    train_loader, val_loader, test_loader = get_dataloader("../Datasets/IMDB_500.csv", [0.7,0.15,0.15], [32,32,32])
    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            texts, labels = batch
            labels = labels.to(device)
            # Tokenize texts
            encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            input_ids = encoded_inputs['input_ids'].to(device)
            attention_mask = encoded_inputs['attention_mask'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate accuracy and average loss
        average_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                texts, labels = batch
                labels = labels.to(device)

                # Tokenize texts
                encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
                input_ids = encoded_inputs['input_ids'].to(device)
                attention_mask = encoded_inputs['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total_samples += labels.size(0)

        val_accuracy = val_correct / val_total_samples
        average_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Val Loss: {average_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Testing
    model.eval()
    test_correct = 0
    test_total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            texts, labels = batch
            labels = labels.to(device)

            # Tokenize texts
            encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            input_ids = encoded_inputs['input_ids'].to(device)
            attention_mask = encoded_inputs['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, predicted = torch.max(logits, 1)
            test_correct += (predicted == labels).sum().item()
            test_total_samples += labels.size(0)

    test_accuracy = test_correct / test_total_samples
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    accuracies.append(test_accuracy)

test_accuracies = np.array(accuracies)
mean_accuracy = np.mean(test_accuracies)
std_accuracy = np.std(test_accuracies)

print("Test Accuracies:")
print(test_accuracies)
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation: {std_accuracy:.4f}")
