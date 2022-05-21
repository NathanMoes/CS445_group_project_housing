import csv
import numpy as np
import math


if __name__ == "__main__":
    train = []
    test = []
    with open('test.csv', newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            test.append(row)
    with open('train.csv', newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            train.append(row)
    print("testData")
    print(test[1])
    print("trainData")
    print(train[1])
    print("testing")
