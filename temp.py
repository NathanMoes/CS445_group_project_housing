import csv
import numpy as np
import math




def pre_format(train, test, classes): 
    


def get_norm_dist(x, u, a):  # gets the norm distribution for the passed in data
    part_one = 0.0  # broken into 2 parts just to make it more readable and easier
    part_two = 0.0  # 2nd part
    part_one = (1 / (math.sqrt(2 * math.pi)))
    part_two = -1 * (((x-u) * (x-u))/(2*a*a))
    result = part_one * math.exp(part_two)  # raise e to power of part two
    if result < 0.0001:  # to not divide by 0 etc
        return 0.0001
    return result


# just a paste in from my program for now, will update for stuff later. So ignore for now
# ignore for now, just has stuff in it from my prog 2. Just pasted in code
# updating it rn, will take some time
def get_standard(input_array, classes):  # gets the standard distribution for the data passed in
    mean = np.zeros((int(classes), 81))  # mean, make it mean for each class by all variables
    std = np.zeros((int(classes), 81))  # standard dev
    type_of = 0  # check what class type it is
    hold = 0.0  # for doing squares, just like it to be self * self.
    for j in range(0, len(input_array)):  # for all rows/data in input calculate
        type_of = int(input_array[j][57])
        if type_of == 1:  # check what class
            for h in range(0, 58):
                mean[0][h] += float(input_array[j][h])  # add in each datum for part
        else:
            for h in range(0, 58):
                mean_not[0][h] += float(input_array[j][h])
    for index in range(0, 58):  # calc mean by doing sum of above div the data number of
        mean[0][index] /= len(input_array)
        mean_not[0][index] /= len(input_array)
    for j in range(0, len(input_array)):  # calc standard deviation
        for h in range(0, 58):  # for spam
            hold = (float(input_array[j][h]) - float(mean[0][h]))
            std[0][h] += (hold * hold)
        for h in range(0, 58):  # for not spam
            hold = (float(input_array[j][h]) - float(mean_not[0][h]))
            std_not[0][h] += (hold * hold)
    for index in range(0, 58):  # finish above calc
        std[0][index] = math.sqrt((std[0][index]) / len(input_array))
        if std[0][index] == 0 or std[0][index] == 0.0:
            std[0][index] = 0.0001
        std_not[0][index] = math.sqrt((std_not[0][index]) / len(input_array))
        if std_not[0][index] == 0 or std_not[0][index] == 0.0:
            std_not[0][index] = 0.0001
    return mean, std, mean_not, std_not  # return arrays of new calc data


if __name__ == "__main__":  # main function call
    train = [] # traindata
    test = [] # testdata
    with open('test.csv', newline='\n') as csvfile:  # read in from test.csv for test data
        csv_reader = csv.reader(csvfile, delimiter=',') # reads in 
        for row in csv_reader:  # append stuff from each row/data point
            test.append(row)  # append
    with open('train.csv', newline='\n') as csvfile: # same as above but for train
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            train.append(row)
    sum_var = 0.0
    maxim = 0.0
    minim = 100000000.0
    temp = 0.0
    for row in range(1, len(train)):
        temp = float(train[row][80])
        sum_var += temp
        if temp < minim:
            minim = temp
        if temp > maxim:
            maxim = temp
    sum_var /= len(train)
    print("AVERAGE = " + str(sum_var))
    print("MAX = " + str(maxim))
    print("MIM = " + str(minim))
    print("testData")
    print(test[1])
    print("trainData")
    print(train[1])
    print("testing")
