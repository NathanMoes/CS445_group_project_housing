import csv
import numpy as np
import math


def get_class(to_check):
    input_v = int(to_check)
    if input_v <= 50000:
        return 0
    if 50000 < input_v <= 100000:
        return 1
    if 100000 < input_v <= 150000:
        return 2
    if 150000 < input_v <= 200000:
        return 3
    if 200000 < input_v <= 250000:
        return 4
    if input_v > 250000:
        return 5


def pre_format(train, test, classes):
    # format the data
    for row in range(0, 1459):
        if train[row][1] == '20':
            train[row][1] = 0
        if train[row][1] == '30':
            train[row][1] = 1
        if train[row][1] == '40':
            train[row][1] = 2
        if train[row][1] == '45':
            train[row][1] = 3
        if train[row][1] == '50':
            train[row][1] = 4
        if train[row][1] == '60':
            train[row][1] = 5
        if train[row][1] == '70':
            train[row][1] = 6
        if train[row][1] == '75':
            train[row][1] = 7
        if train[row][1] == '80':
            train[row][1] = 8
        if train[row][1] == '85':
            train[row][1] = 9
        if train[row][1] == '90':
            train[row][1] = 10
        if train[row][1] == '120':
            train[row][1] = 11
        if train[row][1] == '150':
            train[row][1] = 12
        if train[row][1] == '160':
            train[row][1] = 13
        if train[row][1] == '180':
            train[row][1] = 14
        if train[row][1] == '190':
            train[row][1] = 15
        if test[row][1] == '20':
            test[row][1] = 0
        if test[row][1] == '30':
            test[row][1] = 1
        if test[row][1] == '40':
            test[row][1] = 2
        if test[row][1] == '45':
            test[row][1] = 3
        if test[row][1] == '50':
            test[row][1] = 4
        if test[row][1] == '60':
            test[row][1] = 5
        if test[row][1] == '70':
            test[row][1] = 6
        if test[row][1] == '75':
            test[row][1] = 7
        if test[row][1] == '80':
            test[row][1] = 8
        if test[row][1] == '85':
            test[row][1] = 9
        if test[row][1] == '90':
            test[row][1] = 10
        if test[row][1] == '120':
            test[row][1] = 11
        if test[row][1] == '150':
            test[row][1] = 12
        if test[row][1] == '160':
            test[row][1] = 13
        if test[row][1] == '180':
            test[row][1] = 14
        if test[row][1] == '190':
            test[row][1] = 15
        if train[row][2] == 'A':
            train[row][2] = 0
        if train[row][2] == 'C':
            train[row][2] = 1
        if train[row][2] == 'FV':
            train[row][2] = 2
        if train[row][2] == 'I':
            train[row][2] = 3
        if train[row][2] == 'RH':
            train[row][2] = 4
        if train[row][2] == 'RL':
            train[row][2] = 5
        if train[row][2] == 'RP':
            train[row][2] = 6
        if train[row][2] == 'RM':
            train[row][2] = 7
        if test[row][2] == 'A':
            test[row][2] = 0
        if test[row][2] == 'C':
            test[row][2] = 1
        if test[row][2] == 'FV':
            test[row][2] = 2
        if test[row][2] == 'I':
            test[row][2] = 3
        if test[row][2] == 'RH':
            test[row][2] = 4
        if test[row][2] == 'RL':
            test[row][2] = 5
        if test[row][2] == 'RP':
            test[row][2] = 6
        if test[row][2] == 'RM':
            test[row][2] = 7
        # [3] call normalisation fucnto in eval
        # [4]
        if test[row][5] == 'Pave':
            test[row][5] = 0
        if test[row][5] == 'Grvl':
            test[row][5] = 1
        if train[row][5] == 'Pave':
            train[row][5] = 0
        if train[row][5] == 'Grvl':
            train[row][5] = 1
        if train[row][6] == 'Grvl':
            train[row][6] = 0
        if train[row][6] == 'Pave':
            train[row][6] = 1
        if train[row][6] == 'NA':
            train[row][6] = 2
        if test[row][6] == 'Grvl':
            test[row][6] = 0
        if test[row][6] == 'Pave':
            test[row][6] = 1
        if test[row][6] == 'NA':
            test[row][6] = 2
        if test[row][7] == 'Reg':
            test[row][7] = 0
        if test[row][7] == 'IR1':
            test[row][7] = 1
        if test[row][7] == 'IR2':
            test[row][7] = 2
        if test[row][7] == 'IR3':
            test[row][7] = 3
        if train[row][7] == 'Reg':
            train[row][7] = 0
        if train[row][7] == 'IR1':
            train[row][7] = 1
        if train[row][7] == 'IR2':
            train[row][7] = 2
        if train[row][7] == 'IR3':
            train[row][7] = 3
        if train[row][8] == 'Lvl':
            train[row][8] = 0
        if train[row][8] == 'Bnk':
            train[row][8] = 1
        if train[row][8] == 'HLS':
            train[row][8] = 2
        if train[row][8] == 'Low':
            train[row][8] = 3
        if test[row][8] == 'Lvl':
            test[row][8] = 0
        if test[row][8] == 'Bnk':
            test[row][8] = 1
        if test[row][8] == 'HLS':
            test[row][8] = 2
        if test[row][8] == 'Low':
            test[row][8] = 3
        if test[row][9] == 'AllPub':
            test[row][9] = 0
        if test[row][9] == 'NoSewr':
            test[row][9] = 1
        if test[row][9] == 'NoSeWa':
            test[row][9] = 2
        if test[row][9] == 'ELO':
            test[row][9] = 3
        if train[row][9] == 'AllPub':
            train[row][9] = 0
        if train[row][9] == 'NoSewr':
            train[row][9] = 1
        if train[row][9] == 'NoSeWa':
            train[row][9] = 2
        if train[row][9] == 'ELO':
            train[row][9] = 3
        if train[row][10] == 'Inside':
            train[row][10] = 0
        if train[row][10] == 'Corner':
            train[row][10] = 1
        if train[row][10] == 'CulDSac':
            train[row][10] = 2
        if train[row][10] == 'FR2':
            train[row][10] = 3
        if train[row][10] == 'FR3':
            train[row][10] = 4
        if test[row][10] == 'Inside':
            test[row][10] = 0
        if test[row][10] == 'Corner':
            test[row][10] = 1
        if test[row][10] == 'CulDSac':
            test[row][10] = 2
        if test[row][10] == 'FR2':
            test[row][10] = 3
        if test[row][10] == 'FR3':
            test[row][10] = 4
        if test[row][11] == 'Gtl':
            test[row][11] = 0
        if test[row][11] == 'Mod':
            test[row][11] = 1
        if test[row][11] == 'Sev':
            test[row][11] = 2
        if train[row][11] == 'Gtl':
            train[row][11] = 0
        if train[row][11] == 'Mod':
            train[row][11] = 1
        if train[row][11] == 'Sev':
            train[row][11] = 2
        # 25
        if train[row][12] == 'Blmngtn':
            train[row][12] = 0
        if test[row][12] == 'Blmngtn':
            test[row][12] = 0
        if train[row][12] == 'Blueste':
            train[row][12] = 1
        if test[row][12] == 'Blueste':
            test[row][12] = 1
        if train[row][12] == 'BrDale':
            train[row][12] = 2
        if test[row][12] == 'BrDale':
            test[row][12] = 2
        if train[row][12] == 'BrkSide':
            train[row][12] = 3
        if test[row][12] == 'BrkSide':
            test[row][12] = 3
        if train[row][12] == 'ClearCr':
            train[row][12] = 4
        if test[row][12] == 'ClearCr':
            test[row][12] = 4
        if train[row][12] == 'CollgCr':
            train[row][12] = 5
        if test[row][12] == 'CollgCr':
            test[row][12] = 5
        if train[row][12] == 'Crawfor':
            train[row][12] = 6
        if test[row][12] == 'Crawfor':
            test[row][12] = 6
        if train[row][12] == 'Edwards':
            train[row][12] = 7
        if test[row][12] == 'Edwards':
            test[row][12] = 7
        if train[row][12] == 'Gilbert':
            train[row][12] = 8
        if test[row][12] == 'Gilbert':
            test[row][12] = 8
        if train[row][12] == 'IDOTRR':
            train[row][12] = 9
        if test[row][12] == 'IDOTRR':
            test[row][12] = 9
        if train[row][12] == 'MeadowV':
            train[row][12] = 10
        if test[row][12] == 'MeadowV':
            test[row][12] = 10
        if train[row][12] == 'Mitchel':
            train[row][12] = 11
        if test[row][12] == 'Mitchel':
            test[row][12] = 11
        if train[row][12] == 'Names':
            train[row][12] = 12
        if test[row][12] == 'Names':
            test[row][12] = 12
        if train[row][12] == 'NoRidge':
            train[row][12] = 13
        if test[row][12] == 'NoRidge':
            test[row][12] = 13
        if train[row][12] == 'NPkVill':
            train[row][12] = 14
        if test[row][12] == 'NPkVill':
            test[row][12] = 14
        if train[row][12] == 'NridgHt':
            train[row][12] = 15
        if test[row][12] == 'NridgHt':
            test[row][12] = 15
        if train[row][12] == 'NWAmes':
            train[row][12] = 16
        if test[row][12] == 'NWAmes':
            test[row][12] = 16
        if train[row][12] == 'OldTown':
            train[row][12] = 17
        if test[row][12] == 'OldTown':
            test[row][12] = 17
        if train[row][12] == 'SWISU':
            train[row][12] = 18
        if test[row][12] == 'SWISU':
            test[row][12] = 18
        if train[row][12] == 'Sawyer':
            train[row][12] = 19
        if test[row][12] == 'Sawyer':
            test[row][12] = 19
        if train[row][12] == 'SawyerW':
            train[row][12] = 20
        if test[row][12] == 'SawyerW':
            test[row][12] = 20
        if train[row][12] == 'Somerst':
            train[row][12] = 21
        if test[row][12] == 'Somerst':
            test[row][12] = 21
        if train[row][12] == 'StoneBr':
            train[row][12] = 22
        if test[row][12] == 'StoneBr':
            test[row][12] = 22
        if train[row][12] == 'Timber':
            train[row][12] = 23
        if test[row][12] == 'Timber':
            test[row][12] = 23
        if train[row][12] == 'Veenker':
            train[row][12] = 24
        if test[row][12] == 'Veenker':
            test[row][12] = 24
        if test[row][13] == 'Artery':
            test[row][13] = 0
        if train[row][13] == 'Artery':
            train[row][13] = 0
        if test[row][13] == 'Feedr':
            test[row][13] = 1
        if train[row][13] == 'Feedr':
            train[row][13] = 1
        if test[row][13] == 'Norm':
            test[row][13] = 2
        if train[row][13] == 'Norm':
            train[row][13] = 2
        if test[row][13] == 'RRNn':
            test[row][13] = 3
        if train[row][13] == 'RRNn':
            train[row][13] = 3
        if test[row][13] == 'RRAn':
            test[row][13] = 4
        if train[row][13] == 'RRAn':
            train[row][13] = 4
        if test[row][13] == 'PosN':
            test[row][13] = 5
        if train[row][13] == 'PosN':
            train[row][13] = 5
        if test[row][13] == 'PosA':
            test[row][13] = 6
        if train[row][13] == 'PosA':
            train[row][13] = 6
        if test[row][13] == 'RRNe':
            test[row][13] = 7
        if train[row][13] == 'RRNe':
            train[row][13] = 7
        if test[row][13] == 'RRAe':
            test[row][13] = 8
        if train[row][13] == 'RRAe':
            train[row][13] = 8
        if train[row][14] == 'Artery':
            train[row][14] = 0
        if test[row][14] == 'Artery':
            test[row][14] = 0
        if train[row][14] == 'Feedr':
            train[row][14] = 1
        if test[row][14] == 'Feedr':
            test[row][14] = 1
        if train[row][14] == 'Norm':
            train[row][14] = 2
        if test[row][14] == 'Norm':
            test[row][14] = 2
        if train[row][14] == 'RRNn':
            train[row][14] = 3
        if test[row][14] == 'RRNn':
            test[row][14] = 3
        if train[row][14] == 'RRAn':
            train[row][14] = 4
        if test[row][14] == 'RRAn':
            test[row][14] = 4
        if train[row][14] == 'PosN':
            train[row][14] = 5
        if test[row][14] == 'PosN':
            test[row][14] = 5
        if train[row][14] == 'PosA':
            train[row][14] = 6
        if test[row][14] == 'PosA':
            test[row][14] = 6
        if train[row][14] == 'RRNe':
            train[row][14] = 7
        if test[row][14] == 'RRNe':
            test[row][14] = 7
        if train[row][14] == 'RRAe':
            train[row][14] = 8
        if test[row][14] == 'RRAe':
            test[row][14] = 8
        if train[row][15] == '1Fam':
            train[row][15] = 0
        if test[row][15] == '1Fam':
            test[row][15] = 0
        if train[row][15] == '2FmCon':
            train[row][15] = 1
        if test[row][15] == '2FmCon':
            test[row][15] = 1
        if train[row][15] == 'Duplx':
            train[row][15] = 2
        if test[row][15] == 'Duplx':
            test[row][15] = 2
        if train[row][15] == 'TwnhsE':
            train[row][15] = 3
        if test[row][15] == 'TwnhsE':
            test[row][15] = 3
        if train[row][15] == 'TwnhsI':
            train[row][15] = 4
        if test[row][15] == 'TwnhsI':
            test[row][15] = 4
        if test[row][16] == '1Story':
            test[row][16] = 0
        if train[row][16] == '1Story':
            train[row][16] = 0
        if test[row][16] == '1.5Fin':
            test[row][16] = 1
        if train[row][16] == '1.5Fin':
            train[row][16] = 1
        if test[row][16] == '1.5Unf':
            test[row][16] = 2
        if train[row][16] == '1.5Unf':
            train[row][16] = 2
        if test[row][16] == '2Story':
            test[row][16] = 3
        if train[row][16] == '2Story':
            train[row][16] = 3
        if test[row][16] == '2.5Fin':
            test[row][16] = 4
        if train[row][16] == '2.5Fin':
            train[row][16] = 4
        if test[row][16] == '2.5Unf':
            test[row][16] = 5
        if train[row][16] == '2.5Unf':
            train[row][16] = 5
        if test[row][16] == 'SFoyer':
            test[row][16] = 6
        if train[row][16] == 'SFoyer':
            train[row][16] = 6
        if test[row][16] == 'SLvl':
            test[row][16] = 7
        if train[row][16] == 'SLvl':
            train[row][16] = 7
        if test[row][17] == '':
            test[row][17] = 0
        if train[row][17] == '':
            train[row][17] = 0
    return


def get_norm_dist(x, u, a):  # gets the norm distribution for the passed in data # x - datapoint, u - avg, a - std
    part_one = 0.0  # broken into 2 parts just to make it more readable and easier
    part_two = 0.0  # 2nd part
    part_one = (1 / (math.sqrt(2 * math.pi)))
    part_two = -1 * (((x-u) * (x-u))/(2*a*a))
    result = part_one * math.exp(part_two)  # raise e to power of part two
    if result < 0.0001:  # to not divide by 0 etc
        return 0.0001
    return result


def get_prob(train_v, input_v, class_type, feature_type):
    count = 0
    count_in_class = 0
    number_of = len(train_v)
    for row_t in range(1, number_of):
        if get_class(train_v[row_t][80]) == class_type:
            count_in_class += 1
            if train_v[row_t][feature_type] == input_v[feature_type]:
                count += 1
    probability = float(count/count_in_class)
    if probability <= 0.0001:
        return 0.0001
    return probability


# just a paste in from my program for now, will update for stuff later. So ignore for now
# ignore for now, just has stuff in it from my prog 2. Just pasted in code
# updating it rn, will take some time
def get_standard(input_array, classes):  # gets the standard distribution for the data passed in
    mean = np.zeros((int(classes), 81))  # mean, make it mean for each class by all variables
    std = np.zeros((int(classes), 81))  # standard dev
    type_of = 0  # check what class type it is
    hold = 0.0  # for doing squares, just like it to be self * self.
    """
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
    """


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
    pre_format(test, train, 1)
    print("AVERAGE = " + str(sum_var))
    print("MAX = " + str(maxim))
    print("MIM = " + str(minim))
    print("testData")
    # for taco in range(0, 1300):
        # print(test[taco])
    print("trainData")
    print(train[1])
    print("testing")
    print(get_prob(train, test[1], 0, 1))
    # prob = np.zeros((6, 1))

    # for iteration in range(0, len(test)):
    for iteration in range(1, len(test)):
        maximum = -100000000.0
        max_index = 0
        prob = np.zeros((6, 1))
        for feature in range(0, 80):
            prob[0] += math.log(get_prob(train, test[iteration], 0, feature))
            prob[1] += math.log(get_prob(train, test[iteration], 1, feature))
            prob[2] += math.log(get_prob(train, test[iteration], 2, feature))
            prob[3] += math.log(get_prob(train, test[iteration], 3, feature))
            prob[4] += math.log(get_prob(train, test[iteration], 4, feature))
            prob[5] += math.log(get_prob(train, test[iteration], 5, feature))
        for check in range(0, 6):
            if prob[check] > maximum:
                maximum = prob[check]
                max_index = check
        print("computed gen: " + str(iteration))
        if max_index == 0:
            print("predicted sale price is less than 50k")
        if max_index == 1:
            print("predicted sale price is between 50k and 100k")
        if max_index == 2:
            print("predicted sale price is between 100k and 150k")
        if max_index == 3:
            print("predicted sale price is between 150k and 200k")
        if max_index == 4:
            print("predicted sale price is between 200k and 250k")
        if max_index == 5:
            print("predicted sale price is 250k+")
            # prob[0] += math.log(get_prob(train, test[iteration], 0, feature))
