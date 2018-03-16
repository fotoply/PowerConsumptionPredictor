import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


def stripDate(s):
    time = str(s).split(" ")[1].replace("'", "")
    hour = time.split(":")[0]
    minute = int(time.split(":")[1]) / 60
    return int(hour) + minute


def loadDataFromCSV(filePath):
    return np.loadtxt(fname=filePath, delimiter=",", skiprows=1, usecols=(0, 1, 2), converters={0: stripDate})


def extract_from_data(data):
    power = []
    keys = []
    temperature = []

    power_sum = {}
    power_count = {}
    temp_sum = {}
    temp_count = {}
    for entry in data:
        time = entry[0]
        temp = entry[1]
        power_val = entry[2]

        if time in temp_sum:
            temp_sum[time] += temp
            temp_count[time] += 1
        else:
            temp_sum[time] = temp
            temp_count[time] = 1

        if temp in power_sum:
            power_sum[temp] += power_val
            power_count[temp] += 1
        else:
            power_sum[temp] = power_val
            power_count[temp] = 1

    average_power = []
    for key in power_sum:
        average_power.append(power_sum[key]/power_count[key])

    average_temperature = []
    for key in temp_sum:
        average_temperature.append(temp_sum[key]/temp_count[key])
        keys.append(key)

    average_power = [x for _, x in sorted(zip(keys, average_power))]
    average_temperature = [x for _, x in sorted(zip(keys, average_temperature))]
    keys = sorted(keys)

    return np.array(keys), np.array(average_temperature), np.array(average_power)

def extractTimeAndTemperature(data):
    power = []
    temperature = []
    keys = []
    for entry in data:
        time = entry[0]
        temp = entry[1]
        powerVal = entry[2]

        power.append(powerVal)
        temperature.append(temp)
        keys.append(time)

    return np.array(keys), np.array(temperature), np.array(power)

def run():
    data = loadDataFromCSV("../../data/building1retail.csv")
    train_keys, train_temperature, train_power = extractTimeAndTemperature(data[:-500])#extract_from_data(data[:500])
    test_keys, test_temperature, test_power = extractTimeAndTemperature(data[-500:])#extract_from_data(data[500:600])

    x = []
    for i in range(24):
        x.append(i)
        x.append(i + .25)
        x.append(i + .5)
        x.append(i + .75)

    x = np.array(x)
    x = x[:, np.newaxis]

    plt.figure()

    colors = ['teal', 'yellowgreen', 'gold', 'red']
    for count, degree in enumerate([3, 4, 5]):
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(train_keys[:, np.newaxis], train_temperature[:])
        y_plot = model.predict(x)
        plt.plot(x, y_plot, color=colors[count], linewidth=2, label="degree %d" % degree)
        #print('RMSE', np.sqrt(mean_squared_error(test_temperature, y_plot)))
        #print('R2', r2_score(test_temperature, y_plot))

    #plt.plot(x, y_plot, color='teal', linewidth=2)
    plt.scatter(train_keys, train_temperature, c="k", label="training samples")
    # plt.plot(x, y_pred, c="g", label="KNearest", linewidth=2)
    plt.legend()
    plt.show()


run()
