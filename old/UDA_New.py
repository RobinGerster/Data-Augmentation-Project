import torch
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import nlpaug.augmenter.word as naw
import numpy as np
accuracies = []
for i in range(10):
    # Constants
    model_name = "bert-base-uncased"
    num_labels = 2
    batch_size = 8
    num_epochs = 10
    lr = 3e-5
    uda_coeff = 1.0  # The coefficient for UDA loss
    patience = 3

    # Setting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset = load_dataset("imdb")

    # Shuffle the datasets
    train_dataset = dataset['train'].shuffle().select(range(20))
    val_dataset = dataset['test'].shuffle().select(range(1000))  # select more for splitting
    test_dataset = dataset['test'].shuffle().select(range(1000))  # select more for splitting
    unsup_dataset = dataset["unsupervised"].shuffle().select(range(1000))

    # Initialize tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model = model.to(device)


    # Tokenize function
    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    def augment_text(text):
        aug = naw.SynonymAug(aug_src='wordnet')
        augmented_text = aug.augment(text)
        augmented_text = ' '.join(augmented_text)  # Join the list of strings into a single string
        return augmented_text

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    unsup_dataloader = DataLoader(unsup_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_consistency_loss = 0  # Track consistency loss

        for batch, unsup_batch in zip(train_loader, unsup_dataloader):
            # Labeled data
            texts = batch['text']
            labels = batch['label'].to(device)

            # Tokenize labeled data
            encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            input_ids = encoded_inputs['input_ids'].to(device)
            attention_mask = encoded_inputs['attention_mask'].to(device)

            # Unsupervised data
            unsup_texts = unsup_batch['text']
            unsup_augmented_texts = [augment_text(text) for text in unsup_texts]

            # Tokenize unsupervised data
            unsup_encoded_inputs = tokenizer(unsup_texts, padding=True, truncation=True, return_tensors='pt')
            unsup_input_ids = unsup_encoded_inputs['input_ids'].to(device)
            unsup_attention_mask = unsup_encoded_inputs['attention_mask'].to(device)

            # Tokenize augmented unsupervised data
            unsup_aug_encoded_inputs = tokenizer(unsup_augmented_texts, padding=True, truncation=True, return_tensors='pt')
            unsup_aug_input_ids = unsup_aug_encoded_inputs['input_ids'].to(device)
            unsup_aug_attention_mask = unsup_aug_encoded_inputs['attention_mask'].to(device)

            optimizer.zero_grad()

            # Labeled data forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            supervised_loss = criterion(logits, labels)

            # Unsupervised data forward pass
            unsup_outputs = model(unsup_input_ids, attention_mask=unsup_attention_mask)
            unsup_logits = unsup_outputs.logits
            unsup_probs = F.softmax(unsup_logits, dim=1)

            # Augmented unsupervised data forward pass
            unsup_aug_outputs = model(unsup_aug_input_ids, attention_mask=unsup_aug_attention_mask)
            unsup_aug_logits = unsup_aug_outputs.logits
            unsup_aug_probs = F.softmax(unsup_aug_logits, dim=1)

            # Consistency loss
            consistency_loss = F.kl_div(unsup_probs.log_softmax(dim=1), unsup_aug_probs.softmax(dim=1), reduction='batchmean')

            # Total loss
            final_loss = supervised_loss + uda_coeff * consistency_loss

            # Backward pass and optimization
            final_loss.backward()
            optimizer.step()


        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text']
                labels = batch['label'].to(device)

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
            texts = batch['text']
            labels = batch['label'].to(device)

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