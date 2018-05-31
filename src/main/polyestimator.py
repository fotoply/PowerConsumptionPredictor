import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def stripDate(s):
    time = str(s).split(" ")[1].replace("'", "")
    hour = time.split(":")[0]
    minute = int(time.split(":")[1]) / 60
    return int(hour) + minute


def loadDataFromCSV(filePath):
    return np.loadtxt(fname=filePath, delimiter=",", skiprows=1, usecols=(0, 1, 2), converters={0: stripDate})


def run():
    data = loadDataFromCSV("../../data/Building-1/building1retail.csv")
    trainingData = data[:-2000]
    testData = data[-2000:]
    trainingKey, trainingAvg = extractQuarterlyAverage(trainingData)
    testKey, testAvg = extractQuarterlyAverage(testData)

    y = f(np.array(trainingAvg))

    plt = plotScatter(testKey, testAvg)

    colors = ['teal', 'yellowgreen', 'gold']
    lw = 2
    for count, degree in enumerate([3, 4, 5]):
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(trainingKey[:, np.newaxis], trainingAvg)
        y_plot = model.predict(testKey[:, np.newaxis])
        plt.plot(testKey, y_plot,
                 #color=colors[count],
                 linewidth=lw, label="degree %d" % degree)
        from sklearn.metrics import r2_score
        print("R2 score %d degree: " % degree + str(r2_score(testAvg, y_plot)))

    plt.show()


def plotScatter(xAxis, yAxis):
    plt.scatter(xAxis, yAxis, color='black')
    #plt.xticks(())
    #plt.yticks(())
    return plt


def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)


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

    powerAvg = [x for _,x in sorted(zip(keys, powerAvg))]
    keys = sorted(keys)

    return np.array(keys), np.array(powerAvg)

run()
