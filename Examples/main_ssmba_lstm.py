from Utils.data_loader import get_imdb_ssmba_dataloaders
from Utils.classifiers import SequenceLSTMClassifier
from Utils.trainer import SupervisedTrainer
import torch
import time
import datetime
import numpy as np

start_time = time.time()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# assert str(device) == "cuda", "GPU device not available. Training will be sllooowwwww."

# Number of augmentations
for naug in [1, 2, 4, 8, 16, 32]:
    print("Naug=" + str(naug))
    ood_performance_list = []
    id_performance_list = []
    # 5-fold cross validation
    for i in [1, 2, 3, 4, 5]:
        print("Cross validation: " + str(i))
        print(f'Current time: {datetime.datetime.now().time()}')

        # Get our model
        lstm = SequenceLSTMClassifier(device, pretrained_model_name="bert-base-uncased", num_labels=2)

        # Get the train dataloader
        train_dataloader, imdb_test_dataloader, sst_test_dataloader = get_imdb_ssmba_dataloaders(
            batch_sizes=[16, 32, 32],
            naug=naug)

        # Initialize additional training parameters
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(lstm.model.parameters(), lr=3e-5)

        # Training

        supervised_trainer = SupervisedTrainer(lstm, criterion, optimizer, train_dataloader, id_performance_list,
                                               epochs=5,
                                               device=device,
                                               val_dataloader=imdb_test_dataloader)
        supervised_trainer.train()
        print("id_performance_list: " + str(id_performance_list))

        print(f'Current time: {datetime.datetime.now().time()}, Execution time:',
              time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        # Out-of-Domain Evaluation

        start_time = time.time()

        ood_performance = supervised_trainer.evaluate(sst_test_dataloader)
        print("ood_performance: " + str(ood_performance))
        ood_performance_list.append(ood_performance)

        print(f'Current time: {datetime.datetime.now().time()}, Execution time:',
              time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print("id_performance_list: " + str(id_performance_list))
    print("Average id_performance: " + str(np.mean(id_performance_list)))
    print("ood_performance_list: " + str(ood_performance_list))
    print("Average ood_performance: " + str(np.mean(ood_performance_list)))
