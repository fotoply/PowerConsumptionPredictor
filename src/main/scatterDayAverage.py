import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def stripDate(s):
    time = str(s).split(" ")[1].replace("'", "")
    hour = time.split(":")[0]
    minute = int(time.split(":")[1])/60
    return int(hour)+minute


def loadDataFromCSV(filePath):
    return np.loadtxt(fname=filePath, delimiter=",", skiprows=1, usecols=(0, 1, 2), converters={0:stripDate})

def run():
    data = loadDataFromCSV("../../data/building1retail.csv")
    keys, powerAvg = extractQuarterlyAverage(data)
    plotScatter(keys, powerAvg).show()


def plotScatter(xAxis, yAxis):
    plt.scatter(xAxis, yAxis, color='black')
    plt.xticks(())
    plt.yticks(())
    return plt


def extractQuarterlyAverage(data):
    powerSum = {}
    count = {}
    for entry in data:
        time = entry[0]
        temp = entry[1]
        power = entry[2]

        if time in powerSum:
            powerSum[time] = powerSum[time] + power
            count[time] += 1
        else:
            powerSum[time] = power
            count[time] = 1
    powerAvg = []
    keys = []
    for key in powerSum:
        powerAvg.append(powerSum[key] / count[key])
        keys.append(key)
    return keys, powerAvg


run()

