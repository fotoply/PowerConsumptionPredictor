import pandas as pandio
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

def loadDataFromCSV(filepath):
    return pandio.read_csv(filepath)

data = loadDataFromCSV("../../data/Building-1/building1retail.csv")
#data.plot(x='Timestamp', y='Power (kW)')
data.plot.kde()

plt.show()
