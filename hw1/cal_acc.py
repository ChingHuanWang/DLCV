import numpy as np
import csv

file_len = 0
correct_len = 0

with open("densenet161_prediction.csv", "r") as f:
	csv_reader = csv.reader(f)
	header = next(csv_reader)
	# print(header)

	for row in csv_reader:
		file_len += 1
		Id = row[0].split("_")[0]
		label = row[1]

		if(Id == label):
			correct_len += 1


print(f"acc = {correct_len / file_len * 100}%")