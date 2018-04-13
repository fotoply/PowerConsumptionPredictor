import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def extract(data):
    power = []
    temperature = []
    keys = []
    for entry in data:
        time = entry[0]
        temp = entry[1]
        power_val = entry[2]

        power.append(power_val)
        temperature.append(temp)
        keys.append(time)

    return np.array(keys), np.array(temperature), np.array(power)


def run():
    data = loadDataFromCSV("../../data/building1retail.csv")
    train_keys, train_temperature, train_power = extract(data[:-500])
    test_keys, test_temperature, test_power = extract(data[-500:])

    x = []
    for i in range(24):
        x.append(i)
        x.append(i + .25)
        x.append(i + .5)
        x.append(i + .75)

    x = np.array(x)
    #x = x[:, np.newaxis]

    model = make_pipeline(PolynomialFeatures(5), Ridge())
    model.fit(train_keys[:, np.newaxis], train_temperature[:])
    y_plot = model.predict(x[:, np.newaxis])

    model.fit(train_keys[:, np.newaxis], train_power[:])
    z_plot = model.predict(x[:, np.newaxis])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #ax.plot_surface(x, y_plot[:, np.newaxis], z_plot[:, np.newaxis])
    ax.plot_trisurf(x, y_plot, z_plot)
    ax.scatter(test_keys, test_temperature, zs=0, zdir='z')
    ax.scatter(test_keys, test_power, zs=70, zdir='y')
    ax.scatter(test_temperature, test_power, zs=0, zdir='x')

    ax.set_xlim(0, 24)
    ax.set_ylim(30, 70)
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (F)')
    ax.set_zlabel('Power')

    plt.show()


run()
