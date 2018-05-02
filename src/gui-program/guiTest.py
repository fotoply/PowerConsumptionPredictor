import random
import sys
import pandas as pd
from PyQt5 import QtWidgets as QtGui
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class Window(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # a figure instance to plot on
        self.figure = Figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        # self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = QtGui.QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        # set the layout
        layout = QtGui.QVBoxLayout()
        # layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def plot(self):
        filename, sortOption = QFileDialog.getOpenFileName(self, "Choose data file", "", "All Files (*);;CSV Files (*.csv)")

        if not filename:
            return

        data = pd.read_csv(filename, parse_dates=['Timestamp'], index_col='Timestamp')

        ts = data["Power (kW)"]
        ts_mean = ts.rolling("7d").mean()
        ts_std = ts.rolling("7d").std()

        # create an axis
        ax = self.figure.add_subplot(211)
        ax.clear()
        ax.plot(ts_mean)
        ax = self.figure.add_subplot(212)
        ax.clear()
        ax.plot(ts_std)

        # refresh canvas
        self.canvas.draw()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
