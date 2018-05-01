import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
data = pd.read_csv("../../data/building1retail.csv", parse_dates=['Timestamp'], index_col='Timestamp')
print(data.head())
print('\n Data Types:')
print(data.dtypes)
print(data.index)


ts = data["Power (kW)"]
ts_log = ts.rolling("7d").mean()

plt.plot(ts_log)

#plt.plot(ts)
plt.show()