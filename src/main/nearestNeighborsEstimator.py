import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


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


def findBestK():
    data = loadDataFromCSV("../../data/Building-1/building1retail.csv")
    keys, power = extractTimeAndTemperature(data)
    keys = keys[:, np.newaxis]

    testData = extractTimeAndTemperature(data[-96:])
    x_test, y_test = testData
    x_test = x_test[:, np.newaxis]

    x = []
    for i in range(24):
        x.append(i)
        x.append(i + .25)
        x.append(i + .5)
        x.append(i + .75)

    x = np.array(x)
    x = x[:, np.newaxis]

    from sklearn.metrics import r2_score
    scores = []
    for i in range(1, 1500, 1):
        regressor = KNeighborsRegressor(n_neighbors=i)
        regressor.fit(keys, power)

        pred = regressor.predict(x)
        scores.append(r2_score(y_test, pred))

    plt.figure()
    plt.plot(scores)
    plt.show()


def run():
    data = loadDataFromCSV("../../data/Building-1/building1retail.csv")
    keys, power = extractTimeAndTemperature(data)
    keys = keys[:, np.newaxis]

    regressor = KNeighborsRegressor(n_neighbors=1080)  # best ≈ 1080
    regressor.fit(keys, power)

    testData = extractTimeAndTemperature(data[2000:2500])
    x_test, y_test = testData
    x_test = x_test[:, np.newaxis]

    x = []
    for i in range(24):
        x.append(i)
        x.append(i+.25)
        x.append(i+.5)
        x.append(i+.75)

    x = np.array(x)
    x = x[:, np.newaxis]
    y_pred = regressor.predict(x)

    plt.figure()
    #plt.scatter(x_test, y_test, c="k", label="training samples")
    plt.plot(x, y_pred, c="g", label="KNearest", linewidth=2)
    plt.legend()
    plt.show()

run()
#findBestK()