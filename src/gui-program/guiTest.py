import random
import sys
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets as QtGui
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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
        self.plotTypeBox.addItems(("Rolling mean", "Rolling STD", "Seasonal decomposition", "Representative day"))
        self.plotTypeBox.currentIndexChanged.connect(self.plotTypeChanged)
        self.plotTypeBox.setEnabled(False)

        self.decompVariableBox = QtGui.QComboBox()
        self.decompVariableBox.hide()
        self.decompType = QtGui.QComboBox()
        self.decompType.addItems(("Additive", "Mutiplicative", "Loess (STL)"))
        self.decompType.hide()
        self.loadPredictionButton = QtGui.QPushButton("Load predictive data")
        self.loadPredictionButton.hide()
        self.loadPredictionButton.clicked.connect(self.doPrediction)

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
        layout.addWidget(self.canvas, 1, 2, 10, 1)
        self.setLayout(layout)

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
                grouped[i] = represe[i].groupby(represe[i].index.time).mean()#lambda x: x.hour + x.minute / 60).mean()

                ax = self.figure.add_subplot(dataCount * 100 + 10 + i + 1)
                ax.plot(grouped[i])
                ax.set_title(data.columns.values[i])

            Window.groups = grouped

            if dataCount == 2:
                rng = np.random.RandomState(1)
                from sklearn.ensemble import AdaBoostRegressor
                from sklearn.tree import DecisionTreeRegressor
                from sklearn import linear_model
                regressor = linear_model.LinearRegression()#AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=20, random_state=rng)
                regressor.fit(represe[0].to_frame().dropna().iloc[:], represe[1].dropna().values)
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                x = []
                for i in range(24):
                    x.append(i)
                    x.append(i + .25)
                    x.append(i + .5)
                    x.append(i + .75)

                x = np.array(x)
                x = x[:, np.newaxis]
                y = regressor.predict(grouped[0].to_frame().dropna().iloc[:])
                ax.plot(x, y, c="g", label="Representative Day", linewidth=2)
                Window.regressor = regressor
                self.loadPredictionButton.show()

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
        data_actual = pd.read_csv(filename.replace(".csv", "-actual.csv"),
                                  parse_dates=[self.timestampFieldBox.currentText()],
                                  index_col=self.timestampFieldBox.currentText(),
                                  delimiter=self.selectDelimiterBox.currentText())

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        y = Window.regressor.predict(data.dropna().iloc[:])
        ax.plot(data.index.values, y, label="Prediction", linewidth=2)
        ax.scatter(data_actual.index.values, data_actual.iloc[:], label="Actual")
        self.canvas.draw()

    def decomposeSeries(self, ts, decompType=None):
        decomp = None
        if decompType is None:
            decompType = self.decompType.currentText()

        if decompType == "Additive":
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomp = seasonal_decompose(ts, model="additive", freq=96)

        elif decompType == "Multiplicative":
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomp = seasonal_decompose(ts, model="multiplicative", freq=96)

        elif decompType == "Loess (STL)":
            from stldecompose import decompose
            decomp = decompose(ts, period=96)

        return decomp


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
