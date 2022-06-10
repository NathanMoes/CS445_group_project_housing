import csv
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


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


def get_norm_dist(x, u, a):  # gets the norm distribution for the passed in data # x - datapoint, u - avg, a - std
    part_one = 0.0  # broken into 2 parts just to make it more readable and easier
    part_two = 0.0  # 2nd part
    part_one = (1 / (math.sqrt(2 * math.pi)))
    part_two = -1 * (((x - u) * (x - u)) / (2 * a * a))
    result = part_one * math.exp(part_two)  # raise e to power of part two
    if result < 0.0001:  # to not divide by 0 etc
        return 0.0001
    return result


def get_prob(train_v, input_v, class_type, feature_type):
    mean = 0.0
    std = 0.0
    hold = 0.0
    count = 0
    count_in_class = 0
    number_of = 729
    if feature_type == 0:
        for rows_in in range(1, number_of):
            if get_class(train_v[rows_in][80]) == class_type:
                count += 1
        probability = float(count / number_of)
        return probability
    if feature_type == (3 or 4 or 27 or 35 or 37 or 38 or 39 or 44 or 47 or 45 or 46 or 63 or 67 or 72 or 68 or 69
                        or 70 or 71 or 76 or 77):
        for mean_calc in range(1, 729):
            if get_class(train_v[mean_calc][80]) == class_type:
                count_in_class += 1
                mean += float(train_v[mean_calc][feature_type])
        mean /= 728
        hold = (float(input_v[feature_type]) - float(mean))
        std += (hold * hold)
        std = math.sqrt(std / count_in_class)
        if std < 0.0001:
            std = 0.0001
        return get_norm_dist(float(input_v[feature_type]), mean, std)
    for row_t in range(1, number_of):
        if get_class(train_v[row_t][80]) == class_type:
            count_in_class += 1
            if train_v[row_t][feature_type] == input_v[feature_type]:
                count += 1
    probability = float(count / count_in_class)
    if probability <= 0.0001:
        return 0.0001
    return probability


if __name__ == "__main__":  # main function call
    conf_mat = np.zeros((6, 6))
    train = []  # traindata
    test = []  # testdata
    with open('test.csv', newline='\n') as csvfile:  # read in from test.csv for test data
        csv_reader = csv.reader(csvfile, delimiter=',')  # reads in
        for row in csv_reader:  # append stuff from each row/data point
            test.append(row)  # append
    with open('train.csv', newline='\n') as csvfile:  # same as above but for train
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            train.append(row)
    sum_var = 0.0
    maxim = 0.0
    minim = 100000000.0
    temp = 0.0
    half_point = 729
    correct_count = 0
    for row in range(0, len(train)):
        for feature in range(0, 80):
            if train[row][feature] == 'NA':
                train[row][feature] = 0
    # for iteration in range(0, len(test)):
    median_zero = []
    median_one = []
    median_two = []
    median_three = []
    median_four = []
    median_five = []
    for iteration in range(1, half_point):
        maximum = -100000000.0
        max_index = 0
        prob = np.zeros((6, 1))
        for feature in range(0, 80):
            prob[0] += math.log(get_prob(train, train[iteration + half_point], 0, feature))
            prob[1] += math.log(get_prob(train, train[iteration + half_point], 1, feature))
            prob[2] += math.log(get_prob(train, train[iteration + half_point], 2, feature))
            prob[3] += math.log(get_prob(train, train[iteration + half_point], 3, feature))
            prob[4] += math.log(get_prob(train, train[iteration + half_point], 4, feature))
            prob[5] += math.log(get_prob(train, train[iteration + half_point], 5, feature))
        for check in range(0, 6):
            if prob[check] > maximum:
                maximum = prob[check]
                max_index = check
        print("computed gen: " + str(iteration))
        if max_index == 0:
            # print("predicted sale price is less than 50k")
            result = float(train[iteration + half_point][80])
            median_zero.append(result)
            if get_class(train[iteration + half_point][80]) == 0:
                # print("CORRECT")
                correct_count += 1
                conf_mat[0][0] += 1
            else:
                conf_mat[get_class(train[iteration + half_point][80])][0] += 1
                # print("INCORRECT")
        if max_index == 1:
            result = float(train[iteration + half_point][80])
            # print("predicted sale price is between 50k and 100k")
            median_one.append(result)
            if get_class(train[iteration + half_point][80]) == 1:
                # print("CORRECT")
                correct_count += 1
                conf_mat[1][1] += 1
            else:
                # print("INCORRECT")
                conf_mat[get_class(train[iteration + half_point][80])][1] += 1
        if max_index == 2:
            result = float(train[iteration + half_point][80])
            median_two.append(result)
            # print("predicted sale price is between 100k and 150k")
            if get_class(train[iteration + half_point][80]) == 2:
                # print("CORRECT")
                correct_count += 1
                conf_mat[2][2] += 1
            else:
                # print("INCORRECT")
                conf_mat[get_class(train[iteration + half_point][80])][2] += 1
        if max_index == 3:
            result = float(train[iteration + half_point][80])
            median_three.append(result)
            # print("predicted sale price is between 150k and 200k")
            if get_class(train[iteration + half_point][80]) == 3:
                # print("CORRECT")
                correct_count += 1
                conf_mat[3][3] += 1
            else:
                # print("INCORRECT")
                conf_mat[get_class(train[iteration + half_point][80])][3] += 1
        if max_index == 4:
            result = float(train[iteration + half_point][80])
            median_four.append(result)
            # print("predicted sale price is between 200k and 250k")
            if get_class(train[iteration + half_point][80]) == 4:
                # print("CORRECT")
                correct_count += 1
                conf_mat[4][4] += 1
            else:
                # print("INCORRECT")
                conf_mat[get_class(train[iteration + half_point][80])][4] += 1
        if max_index == 5:
            result = float(train[iteration + half_point][80])
            # print(result)
            median_five.append(result)
            # print("Medians 5: " + str(statistics.median(median_five)))
            # print("predicted sale price is 250k+")
            if get_class(train[iteration + half_point][80]) == 5:
                # print("CORRECT")
                correct_count += 1
                conf_mat[5][5] += 1
            else:
                # print("INCORRECT")
                conf_mat[get_class(train[iteration + half_point][80])][5] += 1
            # prob[0] += math.log(get_prob(train, test[iteration], 0, feature))

    zero = 0
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    # zero_med = statistics.median(median_one)
    """
    one_med = statistics.median(median_one)
    two_med = statistics.median(median_two)
    three_med = statistics.median(median_three)
    four_med = statistics.median(median_four)
    five_med = statistics.median(median_five)
    """
    one_med = 25000.0
    two_med = 75000.0
    three_med = 125000.0
    four_med = 175000.0
    five_med = 225000.0
    for calc in range(0, 6):
        zero += conf_mat[calc][0]
        one += conf_mat[calc][1]
        two += conf_mat[calc][2]
        three += conf_mat[calc][3]
        four += conf_mat[calc][4]
        five += conf_mat[calc][5]
    sum_num = 0.0
    for calc in range(0, int(one)):
        # store = median_one[calc] - one_med
        store = np.log(median_one[calc]) - np.log(one_med)
        sum_num += (store * store)
    sum_num /= one
    print("RMSE of one class: " + str(np.sqrt(sum_num)))
    sum_num = 0.0
    for calc in range(0, int(two)):
        store = np.log(median_two[calc]) - np.log(two_med)
        sum_num += (store * store)
    sum_num /= two
    print("RMSE of two class: " + str(np.sqrt(sum_num)))
    sum_num = 0.0
    for calc in range(0, int(three)):
        store = np.log(median_three[calc]) - np.log(three_med)
        sum_num += (store * store)
    sum_num /= three
    print("RMSE of three class: " + str(np.sqrt(sum_num)))
    sum_num = 0.0
    for calc in range(0, int(four)):
        store = np.log(median_four[calc]) - np.log(four_med)
        sum_num += (store * store)
    sum_num /= four
    print("RMSE of four class: " + str(np.sqrt(sum_num)))
    sum_num = 0.0
    for calc in range(0, int(five)):
        store = np.log(median_five[calc]) - np.log(five_med)
        sum_num += (store * store)
    sum_num /= five
    print("RMSE of five class: " + str(np.sqrt(sum_num)))
    sum_num = 0.0
    # means_square_error_rooted =

    print("Correct % is: " + str(correct_count / 729))
    print(conf_mat)
    print(median_zero)
    # print("Medians 0: " + str(statistics.median(median_zero)))
    print("Medians 1: " + str(statistics.median(median_one)))
    print("Medians 2: " + str(statistics.median(median_two)))
    print("Medians 3: " + str(statistics.median(median_three)))
    print("Medians 4: " + str(statistics.median(median_four)))
    print("Medians 5: " + str(statistics.median(median_five)))
    df_cm = pd.DataFrame(conf_mat, range(6), range(6))
    sn.set(font_scale = 1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size":30})
    plt.show()
