import random
import sys
from math import sqrt

import pandas as pd
import numpy as np
from PyQt5 import QtWidgets as QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class ExploreWindow(QtGui.QDialog):
    def __init__(self, parent=None, inputFigure=None):
        super(ExploreWindow, self).__init__(parent)
        self.figure = inputFigure

        if self.figure is None:
            return

        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)


class Window(QtGui.QDialog):
    filename = None
    headers = None
    regressor = None
    groups = None

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # a figure instance to plot on
        self.figure = Figure()

        # this is the Canvas Widget that displays the `figure`
        self.canvas = FigureCanvas(self.figure)

        # Just some button connected to `plot` method
        self.selectFileButton = QtGui.QPushButton('Select file')
        self.selectFileButton.clicked.connect(self.selectFile)

        self.selectDelimiterDesc = QtGui.QLabel()
        self.selectDelimiterDesc.setText("Select delimiter for file:")
        self.selectDelimiterBox = QtGui.QComboBox()
        self.delimiters = (",", ";", "|", " ")
        self.selectDelimiterBox.addItems(self.delimiters)

        self.timestampDesc = QtGui.QLabel()
        self.timestampDesc.setText("Select row for timestamp:")
        self.timestampFieldBox = QtGui.QComboBox()
        self.timestampFieldBox.setEnabled(False)

        self.plotButton = QtGui.QPushButton("Plot data")
        self.plotButton.clicked.connect(self.plot)
        self.plotButton.setEnabled(False)

        self.loadButton = QtGui.QPushButton("Load data")
        self.loadButton.clicked.connect(self.loadData)

        self.plotTypeBox = QtGui.QComboBox()
        self.plotTypeBox.addItems(
            ("Rolling mean", "Rolling STD", "Seasonal decomposition", "Representative day", "Input data plot"))
        self.plotTypeBox.currentIndexChanged.connect(self.plotTypeChanged)
        self.plotTypeBox.setEnabled(False)

        self.decompVariableBox = QtGui.QComboBox()
        self.decompVariableBox.hide()
        self.decompType = QtGui.QComboBox()
        self.decompType.addItems(("Additive", "Multiplicative", "Loess (STL)"))
        self.decompType.hide()
        self.loadPredictionButton = QtGui.QPushButton("Load predictive data")
        self.loadPredictionButton.hide()
        self.loadPredictionButton.clicked.connect(self.doPrediction)

        self.exploreButton = QtGui.QPushButton("Explore")
        self.exploreButton.clicked.connect(self.showExplore)

        # set the layout
        layout = QtGui.QGridLayout()
        layout.setSpacing(4)
        layout.addWidget(self.selectFileButton, 1, 1)

        delimLayout = QtGui.QVBoxLayout()
        delimLayout.addWidget(self.selectDelimiterDesc)
        delimLayout.addWidget(self.selectDelimiterBox)
        layout.addLayout(delimLayout, 2, 1)

        timeLayout = QtGui.QVBoxLayout()
        timeLayout.addWidget(self.timestampDesc)
        timeLayout.addWidget(self.timestampFieldBox)
        layout.addLayout(timeLayout, 3, 1)
        layout.addWidget(self.loadButton, 4, 1)

        layout.addWidget(self.plotTypeBox, 6, 1)
        layout.addWidget(self.decompVariableBox, 7, 1)
        layout.addWidget(self.decompType, 8, 1)
        layout.addWidget(self.loadPredictionButton, 9, 1)

        layout.addWidget(self.plotButton, 10, 1)
        layout.addWidget(self.canvas, 1, 2, 9, 1)
        layout.addWidget(self.exploreButton, 10, 2)
        self.setLayout(layout)

    def showExplore(self):
        popup = ExploreWindow(inputFigure=self.figure)
        popup.show()

    def plotTypeChanged(self):
        if self.plotTypeBox.currentIndex() == 2:
            self.decompVariableBox.show()
            self.decompType.show()
        else:
            self.decompVariableBox.hide()
            self.decompType.hide()

        if self.plotTypeBox.currentIndex() == 3:
            self.decompType.show()

        return

    def loadData(self):
        if not Window.filename:
            print("Invalid file")
            return

        Window.data = pd.read_csv(Window.filename, parse_dates=[self.timestampFieldBox.currentText()],
                                  index_col=self.timestampFieldBox.currentText(),
                                  delimiter=self.selectDelimiterBox.currentText())
        self.plotButton.setEnabled(True)
        self.plotTypeBox.setEnabled(True)
        self.decompVariableBox.clear()
        self.decompVariableBox.addItems(list(Window.data.columns.values))

    def selectFile(self):
        filename, sortOption = QFileDialog.getOpenFileName(self, "Choose data file", "",
                                                           "All Files (*);;CSV Files (*.csv)")
        if not filename:
            return

        Window.filename = filename
        Window.headers = list(
            pd.read_csv(filename, nrows=1, delimiter=self.selectDelimiterBox.currentText()).columns.values)
        self.timestampFieldBox.clear()
        self.timestampFieldBox.addItems(self.headers)
        self.timestampFieldBox.setEnabled(True)

    def plot(self):
        print("Plotting data")

        data = Window.data
        self.figure.clear()

        if self.plotTypeBox.currentIndex() == 0:
            ts = data
            ts_mean = ts.rolling("7d").mean()

            # create an axis
            ax = self.figure.add_subplot(111)
            ax.clear()
            ax.plot(ts_mean)
            self.figure.legend(ts.columns, loc="upper left")
        elif self.plotTypeBox.currentIndex() == 1:
            ts = data
            ts_std = ts.rolling("7d").std()

            ax = self.figure.add_subplot(111)
            ax.clear()
            ax.plot(ts_std)
            self.figure.legend(ts.columns, loc="upper left")

        elif self.plotTypeBox.currentIndex() == 2:
            ts = data[self.decompVariableBox.currentText()]  # Get the column we are decomposing
            # ts = ts.resample("H").mean().interpolate("linear")  # Resample to a hourly to minimize sampling size
            decomp = self.decomposeSeries(ts)

            if decomp is None:
                print("Unable to perform seasonality decomposition")
                self.canvas.draw()
                return

            ax = self.figure.add_subplot(411)
            ax.plot(decomp.observed)
            ax.set_ylabel("Observed")
            ax.set_xticklabels(labels=())
            ax = self.figure.add_subplot(412)
            ax.plot(decomp.trend)
            ax.set_ylabel("Trend")
            ax.set_xticklabels(labels=())
            ax = self.figure.add_subplot(413)
            ax.plot(decomp.seasonal)
            ax.set_ylabel("Seasonal")
            ax.set_xticklabels(labels=())
            ax = self.figure.add_subplot(414)
            ax.plot(decomp.resid)
            ax.set_ylabel("Residual")

        elif self.plotTypeBox.currentIndex() == 3:
            dataCount = len(data.columns)
            represe = [None] * dataCount
            grouped = [None] * dataCount
            decomp = [None] * dataCount
            for i in range(0, dataCount):
                ts = data[data.columns[i]]
                decomp[i] = self.decomposeSeries(ts)

                if decomp is None:
                    print("Unable to decompose " + data.columns.values[i])
                    continue

                represe[i] = decomp[i].seasonal + decomp[i].trend
                grouped[i] = represe[i].groupby(represe[i].index.hour + represe[i].index.minute / 60).mean()

                # separate time arguments
                represe[i] = represe[i].to_frame()
                represe[i] = represe[i].join(pd.Series(ts.index.month, name="Month", index=ts.index))
                represe[i] = represe[i].join(pd.Series(ts.index.day, name="Date", index=ts.index))
                represe[i] = represe[i].join(pd.Series(ts.index.hour, name="Hour", index=ts.index))
                represe[i] = represe[i].join(pd.Series(ts.index.minute, name="Minute", index=ts.index))

                grouped[i] = grouped[i].to_frame()
                grouped[i] = grouped[i].join(pd.Series(6, index=grouped[i].index, name="Month"))
                grouped[i] = grouped[i].join(pd.Series(15, index=grouped[i].index, name="Date"))
                grouped[i] = grouped[i].join(pd.Series(grouped[i].index.map(int), index=grouped[i].index, name="Hour"))
                grouped[i] = grouped[i].join(
                    pd.Series(grouped[i].index.map(lambda x: (int(x * 100) - int(x) * 100) / (1 + 2 / 3)),
                              index=grouped[i].index,
                              name="Minute"))

                ax = self.figure.add_subplot(dataCount * 100 + 10 + i + 1)
                ax.plot(grouped[i])
                ax.set_title(data.columns.values[i])

            Window.groups = grouped

            if dataCount == 2:
                rng = np.random.RandomState(1)
                from sklearn.ensemble import AdaBoostRegressor
                from sklearn.tree import DecisionTreeRegressor
                from sklearn import linear_model
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.linear_model import Ridge
                regressor = make_pipeline(PolynomialFeatures(3), DecisionTreeRegressor(
                    max_depth=10))  # linear_model.LinearRegression(fit_intercept=False)  # AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=20, random_state=rng)
                regressor.fit(represe[0].dropna().values, represe[1].dropna().iloc[:, 0])
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                y = regressor.predict(grouped[0].dropna().values)
                print(represe[0].dropna().head(20))
                print(represe[0].dropna().describe())
                print(grouped[0].dropna().head(20))
                print(grouped[0].dropna().describe())
                ax.plot(grouped[0].index.values, y, c="r", label="Representative Day", linewidth=2)
                Window.regressor = regressor
                self.loadPredictionButton.show()
        elif self.plotTypeBox.currentIndex() == 4:
            ax = self.figure.add_subplot(111)
            ax.clear()
            ax.plot(data)
            self.figure.legend(data.columns, loc="upper left")

        # refresh canvas
        self.canvas.draw()

    def doPrediction(self):
        if Window.regressor is None:
            return

        filename, sortOption = QFileDialog.getOpenFileName(self, "Choose data file", "",
                                                           "All Files (*);;CSV Files (*.csv)")
        if not filename:
            return

        data = pd.read_csv(filename, parse_dates=[self.timestampFieldBox.currentText()],
                           index_col=self.timestampFieldBox.currentText(),
                           delimiter=self.selectDelimiterBox.currentText())

        data_actual = None
        from pathlib import Path
        if Path(filename.replace(".csv", "-actual.csv")).is_file():
            data_actual = pd.read_csv(filename.replace(".csv", "-actual.csv"),
                                      parse_dates=[self.timestampFieldBox.currentText()],
                                      index_col=self.timestampFieldBox.currentText(),
                                      delimiter=self.selectDelimiterBox.currentText())

        data = data.join(pd.Series(data.index.month, name="Month", index=data.index))
        data = data.join(pd.Series(data.index.day, name="Date", index=data.index))
        data = data.join(pd.Series(data.index.hour, name="Hour", index=data.index))
        data = data.join(pd.Series(data.index.minute, name="Minute", index=data.index))

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        y = Window.regressor.predict(data.dropna().values)
        ax.plot(data.index.values, y, label="Prediction", linewidth=2)
        y_actual = data_actual.iloc[:]
        if data_actual is not None:
            ax.scatter(data_actual.index.values, y_actual, label="Actual")
        self.canvas.draw()

        y_actual = y_actual.values

        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, average_precision_score, accuracy_score
        squaredError = mean_squared_error(y_actual, y)
        meanError = sqrt(squaredError)
        absoluteError = mean_absolute_error(y_actual, y)
        mape = np.mean(np.abs((y_actual - y) / y_actual)) * 100
        r2Score = r2_score(y_actual, y)
        withinSTD = self.percentInSTD(y, y_actual)
        # precision = average_precision_score(y_actual, y) #These don't work for continous values
        # accuracy = accuracy_score(y_actual, y)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Relevant data for prediction")
        msg.setWindowTitle("Prediction information")
        msg.setDetailedText("Squared mean error: " + str(squaredError) + "\n" +
                            "Mean error: " + str(meanError) + "\n" +
                            "Mean absolute error: " + str(absoluteError) + "\n" +
                            "Mean absolute percentage error: " + str(mape) + "\n" +
                            "R²: " + str(r2Score) + "\n" +
                            "Within standard deviation: " + str(withinSTD) + "%" + "\n")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def decomposeSeries(self, ts, decompType=None):
        decomp = None
        if decompType is None:
            decompType = self.decompType.currentText()

        if decompType == "Additive":
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomp = seasonal_decompose(ts, model="additive", freq=96)

        elif decompType == "Multiplicative":
            from statsmodels.tsa.seasonal import seasonal_decompose
            try:
                decomp = seasonal_decompose(ts, model="multiplicative", freq=96)
            except:
                decomp = None

        elif decompType == "Loess (STL)":
            from stldecompose import decompose
            decomp = decompose(ts, period=96)

        return decomp

    def percentInSTD(self, y, y_actual):
        data = Window.data
        deviation = data.dropna().std()[1]
        inside = 0
        for y_value, y_actual_value in zip(y, y_actual):
            if y_value - deviation < y_actual_value[0] or y_value + deviation > y_actual_value[0]:
                inside += 1
        return inside / len(y) * 100

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
