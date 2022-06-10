# OLS Regression Algorithm for determining House Prices
# For CS 545 Class Project
# By Kirk Jungles

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

### Import Data

# Import data and convert to numpy array
dat_file = 'C:/Users/jungkirk/Documents/School/CS545/project/train.csv'

df = pd.read_csv(dat_file)

### Preprocessing data

#Drop ID column
df = df.drop(columns=['Id'])

#Imported training data with no headers
D = df.values
Dt = np.transpose(D)

### Preprocess Data

# Use sklearn preprocessing module to encode category data into numbers
le = preprocessing.LabelEncoder()

for i, col in enumerate(Dt):
    for element in col:
        #If an element of data column is a string, replace all elements with string of same value and encode
        if type(element) == str:
            col_str = col.astype(str)
            #print(i, col_str)
            col_trans = le.fit_transform(col_str)
            #print(type(col_trans))
            Dt[i] = col_trans
            #print(col_trans)
            break

D = np.transpose(Dt)

#Remove nan's
D = D.astype(float)
D[np.isnan(D)] = 0

### OLS loop(collect data from multiple iterations)

loops = 50

err_RMS = []
err_pcnt = []

for k in range(loops):
    ## Split data into Train and Test, extract prices.

    #Shuffle so that new results are found each iteration
    np.random.shuffle(D)

    #Split data into train and test
    m = (2*len(D))//3
    D_train = D[:m]
    D_test = D[m:]

    D_train_t = np.transpose(D_train)
    D_test_t = np.transpose(D_test)

    #Extract housing prices
    prices_train = D_train_t[-1]
    prices_test = D_test_t[-1]

    #Store data without prices 
    D_train = np.transpose(D_train_t[:-1]) 
    D_train_t = np.transpose(D_train)

    D_test = np.transpose(D_test_t[:-1]) 
    D_test_t = np.transpose(D_test)

    ### OLS Regression

    #With an overdetermined system, we want to solve A x = b, where A is D_train, x is the parameters, and b is prices_train

    #Solution will be of form: x = (A^tA)^-1 A^t b
    #assumes A^tA is singular

    AtA = np.matmul(D_train_t, D_train)

    AtA_inv = np.linalg.pinv(AtA) #pseudoinverse

    At_b = np.matmul(D_train_t, prices_train)

    x = np.matmul(AtA_inv, At_b)

    ## Based on paramaters x, calculate the predicted prices based on D_test
    prices_pred = np.matmul(D_test,x)

    #Calculate Error
    error_percent = np.mean((prices_pred-prices_test)/prices_test*100)
    err_pcnt.append(error_percent)

    #RMSE
    E = np.log(np.abs(prices_pred)) - np.log(prices_test)
    SE = np.square(E)
    MSE = np.mean(SE)
    RMSE = np.sqrt(MSE)
    err_RMS.append(RMSE)


#Calculate mean percent errors over all iterations
mean_pcnt = np.mean(err_pcnt)
print("Mean % Error = ", mean_pcnt)

mean_RMSE = np.mean(err_RMS)
print("Mean RMSE = ", mean_RMSE)

### Plot Results
#Sort 
ind_srt = np.argsort(prices_test)

prices_t_srt = prices_test[ind_srt]
prices_p_srt = prices_pred[ind_srt]

fig_width = 15 #inches
fig_height = 10 #inches
plt.rcParams['figure.figsize'] = [fig_width,fig_height] 
#plt.rcParams["legend.loc"] = 'lower right' 

fig = plt.figure()

plt.plot(prices_t_srt, label='Actual')
plt.plot(prices_p_srt, label='Predicted')
plt.ylabel("Price($)")
plt.xlabel("Index")
plt.title('Price vs Index of Houses')
leg = fig.legend(loc='lower right', ncol=1, borderaxespad=7)
plt.show()
