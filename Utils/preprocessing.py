import csv
import os


def preprocess_imdb_for_ssmba_augmentation():
    path = r'../Datasets/ssmba'
    os.makedirs(path, exist_ok=True)

    with open('../Datasets/IMDB_20k_ssmba_train.csv', "r", encoding="utf8") as csvfile, open(os.path.join(path, "input.txt"), "w", encoding="utf8") as input_file, open(os.path.join(path, "label.txt"), "w") as labels_file:
        csvReader = csv.reader(csvfile, delimiter=',')
        for row in csvReader:
            labels_file.write(row[0] + '\n')
            input_file.write(row[1] + '\n')


def ssmba_augmented_to_csv():
    path = r'../Datasets/ssmba/augmented'
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "ssmba_out.txt"), "r", encoding="utf8") as input_augmented, open(os.path.join(path, "ssmba_out.label"), "r") as labels_augmented, \
            open('../Datasets/IMDB_500_ssmba_augmented_8.csv', "w") as csvfile:
        reader1 = labels_augmented.readlines()
        reader2 = input_augmented.readlines()
        writer = csv.writer(csvfile, lineterminator='\n')

        # Iterate through each line and write the sum to the CSV file
        for line1, line2 in zip(reader1, reader2):
            value1 = int(line1.strip())
            value2 = line2.rstrip().encode("utf-8")
            writer.writerow([value1, value2])


if __name__ == "__main__":
    preprocess_imdb_for_ssmba_augmentation()
    # ssmba_augmented_to_csv()
