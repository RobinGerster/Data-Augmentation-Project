import torch


class SupervisedTrainer:
    def __init__(self, model, criterion, optimizer, train_dataloader, device="gpu", epochs=10, val_dataloader=None,
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

            if self.val_dataloader:
                self.evaluate(self.val_dataloader)

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
