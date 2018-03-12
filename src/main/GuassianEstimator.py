import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def stripDate(s):
    time = str(s).split(" ")[1].replace("'", "")
    hour = time.split(":")[0]
    minute = int(time.split(":")[1]) / 60
    return int(hour) + minute


def loadDataFromCSV(filePath):
    return np.loadtxt(fname=filePath, delimiter=",", skiprows=1, usecols=(0, 1, 2), converters={0: stripDate})


def extractTimeAndTemperature(data):
    power = []
    keys = []
    for entry in data:
        time = entry[0]
        temp = entry[1]
        powerVal = entry[2]

        power.append(powerVal)
        keys.append(time)

    return np.array(keys), np.array(power)





def run():
    data = loadDataFromCSV("../../data/building1retail.csv")
    keys, powerAvg = extractTimeAndTemperature(data[:-30000])
    keys = keys[:, np.newaxis]

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(keys, powerAvg)
    y_pred, sigma = gp.predict(keys, return_std=True)

    plt.figure()
    plt.scatter(keys, powerAvg, c="k", label="training samples")
    plt.plot(keys, y_pred, c="g", label="gaussian", linewidth=2)
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
