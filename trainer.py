import torch
import torch.nn.functional as F
import nlpaug.augmenter.word as naw


class SupervisedTrainer:
    def __init__(self, model, criterion, optimizer, train_dataloader, device="gpu", epochs=1, val_dataloader=None,
                 max_length=512):  # Add max_length as a parameter
        self.tokenizer = model.tokenizer
        self.model = model.model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.epochs = epochs
        self.max_length = max_length  # Store the maximum length

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()

                texts, labels = batch

                # Batch encoding of texts with max_length parameter
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                                        max_length=self.max_length)

                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            avg_train_loss = total_loss / len(self.train_dataloader)
            val_acc = "None"
            if self.val_dataloader is not None:
                val_acc = self.evaluate(self.val_dataloader)
            print(f'Epoch {epoch + 1}/{self.epochs}, Train Loss: {avg_train_loss}, Validation Accuracy: {val_acc}')

    def evaluate(self, dataloader):
        self.model.eval()

        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for batch in dataloader:
                texts, labels = batch

                # Batch encoding of texts with max_length parameter
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                                        max_length=self.max_length)

                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_examples += labels.size(0)

        return total_correct / total_examples


aug = naw.SynonymAug(aug_src='wordnet')


def augment_text(text):
    augmented_text = aug.augment(text)
    augmented_text = ' '.join(augmented_text)  # Join the list of strings into a single string
    return augmented_text


class UDATrainer():
    def __init__(self, model, optimizer, supervised_dataloader, unsupervised_dataloader, device="gpu",
                 epochs=1, val_dataloader=None,
                 max_length=512):  # Add max_length as a parameter
        self.tokenizer = model.tokenizer
        self.model = model.model.to(device)
        self.optimizer = optimizer
        self.unsupervised_dataloader = unsupervised_dataloader
        self.supervised_dataloader = supervised_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.epochs = epochs
        self.max_length = max_length  # Store the maximum length
        self.uda_coeff = 1.0
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch, unsup_batch in zip(self.supervised_dataloader, self.unsupervised_dataloader):
                # Labeled data
                texts, labels = batch
                labels = labels.to(self.device)

                # Tokenize labeled data
                encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                                                max_length=self.max_length)
                input_ids = encoded_inputs['input_ids'].to(self.device)
                attention_mask = encoded_inputs['attention_mask'].to(self.device)

                # Unsupervised data
                unsup_texts, _ = unsup_batch
                unsup_augmented_texts = [augment_text(text) for text in unsup_texts]

                # Tokenize unsupervised data
                unsup_encoded_inputs = self.tokenizer(unsup_texts, padding=True, truncation=True, return_tensors='pt',
                                                      max_length=self.max_length)
                unsup_input_ids = unsup_encoded_inputs['input_ids'].to(self.device)
                unsup_attention_mask = unsup_encoded_inputs['attention_mask'].to(self.device)

                # Tokenize augmented unsupervised data
                unsup_aug_encoded_inputs = self.tokenizer(unsup_augmented_texts, padding=True, truncation=True,
                                                          return_tensors='pt', max_length=self.max_length)
                unsup_aug_input_ids = unsup_aug_encoded_inputs['input_ids'].to(self.device)
                unsup_aug_attention_mask = unsup_aug_encoded_inputs['attention_mask'].to(self.device)

                self.optimizer.zero_grad()

                # Labeled data forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                supervised_loss = self.criterion(logits, labels)

                # Unsupervised data forward pass
                unsup_outputs = self.model(unsup_input_ids, attention_mask=unsup_attention_mask)
                unsup_logits = unsup_outputs.logits
                unsup_probs = F.softmax(unsup_logits, dim=1)

                # Augmented unsupervised data forward pass
                unsup_aug_outputs = self.model(unsup_aug_input_ids, attention_mask=unsup_aug_attention_mask)
                unsup_aug_logits = unsup_aug_outputs.logits
                unsup_aug_probs = F.softmax(unsup_aug_logits, dim=1)

                # Consistency loss
                consistency_loss = F.kl_div(unsup_probs.log_softmax(dim=1), unsup_aug_probs.softmax(dim=1),
                                            reduction='batchmean')

                # Total loss
                combined_loss = supervised_loss + self.uda_coeff * consistency_loss
                total_loss += combined_loss.item()  # Update total_loss

                # Backward pass and optimization
                combined_loss.backward()
                self.optimizer.step()

            # Average loss calculation should be done here
            avg_train_loss = total_loss / len(self.supervised_dataloader)

            # Validation accuracy calculation
            val_acc = "None"
            if self.val_dataloader is not None:
                val_acc = self.evaluate(self.val_dataloader)
            print(f'Epoch {epoch + 1}/{self.epochs}, Train Loss: {avg_train_loss}, Validation Accuracy: {val_acc}')

    def evaluate(self, dataloader):
        self.model.eval()

        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for batch in dataloader:
                texts, labels = batch

                # Batch encoding of texts with max_length parameter
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                                        max_length=self.max_length)

                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_examples += labels.size(0)

        return total_correct / total_examples
