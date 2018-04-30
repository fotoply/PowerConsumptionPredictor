import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

rng = np.random.RandomState(1)
#linear_model.LinearRegression()#
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=20, random_state=rng)
class dataClass:
    reprenX = []
    testX = []
    testY = []

def plotPredicted():
    time = []
    for i in range(24):
        time.append(i)
        time.append(i + .25)
        time.append(i + .5)
        time.append(i + .75)

    time = np.array(time)
    #Test data array
    temp = [50, 50, 50, 50, 50, 51, 52, 51, 51, 50, 50, 50, 50, 49, 49, 49, 48, 48, 47, 46, 47, 47, 47, 48, 49, 49, 49, 49, 49, 50, 50, 50, 52, 52, 53, 53, 53, 52, 51, 51, 50, 51, 52, 53, 54, 53, 54, 54, 54, 54, 55, 56, 57, 57, 57, 57, 56, 57, 58, 58, 58, 59, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 58, 58, 57, 57, 56, 54, 53, 53, 52, 51, 50, 50, 50, 50, 50, 50, 49, 49, 49, 49, 49]

    #For generating random numbers, not very good
    #for i in range(96):
    #    temp.append(random.randint(55, 65))
    temp = np.array(temp)

    X = np.column_stack((time, temp))
    y = regressor.predict(dataClass.testX)

    plt.plot(time, y, c="r", label="Predicted Day", linewidth=2)

    return

def plotRepresentative():

    x = []
    for i in range(24):
        x.append(i)
        x.append(i + .25)
        x.append(i + .5)
        x.append(i + .75)

    x = np.array(x)
    x = x[:, np.newaxis]
    y = regressor.predict(dataClass.reprenX)
    plt.plot(x, y, c="g", label="Representative Day", linewidth=2)

    return

def plotActual():

    x = []
    for i in range(24):
        x.append(i)
        x.append(i + .25)
        x.append(i + .5)
        x.append(i + .75)
    x = np.array(x)
    x = x[:, np.newaxis]
    plt.scatter(x, dataClass.testY, label="Actual data")

    return

def fit():

    data = loadDataFromCSV("../../data/building1retail.csv")
    time, power, temperature = extractTimeAndPower(data)
    X = np.column_stack((time, temperature))
    y = power[:, np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.01, train_size=0.99)
    dataClass.reprenX = X[:96]
    dataClass.testX = X[768:864]
    dataClass.testY = y[768:864]
    regressor.fit(X_train, y_train)

    return

def stripDate(s):
    time = str(s).split(" ")[1].replace("'", "")
    hour = time.split(":")[0]
    minute = int(time.split(":")[1]) / 60
    return int(hour) + minute

def loadDataFromCSV(filePath):
    return np.loadtxt(fname=filePath, delimiter=",", skiprows=1, usecols=(0, 1, 2), converters={0: stripDate})

def extractTimeAndPower(data):
    power = []
    keys = []
    tempe = []
    for entry in data:
        time = entry[0]
        temp = entry[1]
        powerVal = entry[2]

        power.append(powerVal)
        keys.append(time)
        tempe.append(temp)

    return np.array(keys), np.array(power), np.array(tempe)

def run():
    fit()

    plt.figure()
    plotPredicted()
    plotRepresentative()
    plotActual()
    plt.xlabel("Time")
    plt.ylabel("Power consumption")
    plt.title("Comparison")
    plt.legend()
    plt.show()

run()

