import csv
import os

path = r'../Datasets/ssmba'
os.makedirs(path, exist_ok=True)

with open('../Datasets/IMDB_500.csv') as csvfile, open(os.path.join(path, "input.txt"), "w") as input_file, open(os.path.join(path, "label.txt"), "w") as label_file:
    csvReader = csv.reader(csvfile, delimiter=',')
    for row in csvReader:
        label_file.write(row[0] + '\n')
        input_file.write(row[1] + '\n')
