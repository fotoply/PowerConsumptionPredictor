import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def loadDataFromCSV(filePath):
    return np.loadtxt(fname=filePath, delimiter=",", skiprows=1, usecols=(1, 2))

def run():
    data = loadDataFromCSV("../../data/Building-1/building1retail.csv")

    # Split the data into training/testing sets
    temperature = data[:, np.newaxis, 0]
    testSize = int(len(data)/20)
    x_train = temperature[:-testSize]
    x_test = temperature[-testSize:]

    power = data[:, np.newaxis, 1]
    y_train = power[:-testSize]
    y_test = power[-testSize:]

    regressor = linear_model.LinearRegression()
    regressor.fit(x_train, y_train)

    y_prediction = regressor.predict(x_test);

    # The coefficients
    print('Coefficients: \n', regressor.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_prediction))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_prediction))

    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_test, y_prediction, color='blue', linewidth=3)


    plt.show()

run()

