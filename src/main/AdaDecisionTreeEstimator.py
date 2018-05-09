import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


def stripDate(s):
    time = str(s).split(" ")[1].replace("'", "")
    hour = time.split(":")[0]
    minute = int(time.split(":")[1]) / 60
    return int(hour) + minute


def loadDataFromCSV(filePath):
    return np.loadtxt(fname=filePath, delimiter=",", skiprows=1, usecols=(0, 1, 2), converters={0: stripDate})


def run():
    data = loadDataFromCSV("../../data/Building-1/building1retail.csv")
    keys, powerAvg = extractQuarterlyAverage(data)
    keys = keys[:, np.newaxis]
    rng = np.random.RandomState(1)
    regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=10, random_state=rng)
    regressor.fit(keys, powerAvg)
    y = regressor.predict(keys)

    plt.figure()
    plt.scatter(keys, powerAvg, c="k", label="training samples")
    plt.plot(keys, y, c="g", label="n_estimators=10", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show()


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
