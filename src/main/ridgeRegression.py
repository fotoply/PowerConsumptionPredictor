import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score


def stripDate(s):
    time = str(s).split(" ")[1].replace("'", "")
    hour = time.split(":")[0]
    minute = int(time.split(":")[1]) / 60
    return int(hour) + minute


def loadDataFromCSV(filePath):
    return np.loadtxt(fname=filePath, delimiter=",", skiprows=1, usecols=(0, 1, 2), converters={0: stripDate})


def extract(data):
    power = []
    keys = []
    temperature = []
    for entry in data:
        time = entry[0]
        temp = entry[1]
        powerVal = entry[2]

        power.append(powerVal)
        keys.append(time)
        temperature.append(temp)

    return keys, temperature, power


def run():
    data = loadDataFromCSV("../../data/Building-1/building1retail.csv")
    time, tempe, power = extract(data[:5000])
    train_x = np.array(time)[:, np.newaxis]
    train_y = np.array(power)
    time_t, tempe_t, test_y = extract(data[-96:])
    print(train_y)

    x = []
    for i in range(24):
        x.append(i)
        x.append(i + .25)
        x.append(i + .5)
        x.append(i + .75)

    x = np.array(x)
    x = x[:, np.newaxis]

    regressor = linear_model.Ridge(alpha=0.0001, fit_intercept=True)
    regressor.fit(train_x, train_y)
    y_pred = regressor.predict(x)

    print('r2 score: ' + str(r2_score(test_y, y_pred)))

    plt.xlabel('time')
    plt.ylabel('power')
    plt.scatter(train_x, train_y, c='r')
    plt.plot(x, y_pred, c='black', lw=2)
    plt.axis('tight')
    plt.show()


run()
