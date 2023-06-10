import csv
import os
from datasets import load_dataset


def preprocess_imdb_for_ssmba_augmentation():
    path = r'../Datasets/ssmba'
    os.makedirs(path, exist_ok=True)

    with open('../Datasets/IMDB_20k_ssmba_train.csv', "r", encoding="utf8") as csvfile, open(os.path.join(path, "input.txt"), "w", encoding="utf8") as input_file, open(os.path.join(path, "labels.txt"), "w") as labels_file:
        csvReader = csv.reader(csvfile, delimiter=',')
        for row in csvReader:
            labels_file.write(row[0] + '\n')
            input_file.write(row[1] + '\n')


def ssmba_augmented_to_csv():
    path = r'../Datasets/ssmba/augmented'
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "ssmba_out_imdb_500_1.txt"), "r", encoding="utf8") as input_augmented, open(os.path.join(path, "ssmba_out_imdb_500_1.label"), "r") as labels_augmented, \
            open('../Datasets/IMDB_500_1_ssmba_train.csv', "w") as csvfile:
        reader1 = labels_augmented.readlines()
        reader2 = input_augmented.readlines()
        writer = csv.writer(csvfile, lineterminator='\n')

        # Iterate through each line and write the sum to the CSV file
        for line1, line2 in zip(reader1, reader2):
            value1 = int(line1.strip())
            value2 = line2.rstrip().encode("utf-8")
            writer.writerow([value1, value2])


def process_sst2_data():
    val_dataset = load_dataset('glue', 'sst2', split='validation')
    text = val_dataset["sentence"]
    labels = val_dataset["label"]
    with open('../Datasets/SST-2_100_ssmba_test.csv', "w") as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        i = 0
        for sentence, label in zip(text, labels):
            if i == 100:
                break
            writer.writerow([label, sentence.encode("utf-8")])
            i += 1


if __name__ == "__main__":
    process_sst2_data()
    # preprocess_imdb_for_ssmba_augmentation()
    # ssmba_augmented_to_csv()
