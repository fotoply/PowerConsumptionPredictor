import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
data = pd.read_csv("../../data/building1retail.csv", parse_dates=['Timestamp'], index_col='Timestamp')
print(data.head())
print('\n Data Types:')
print(data.dtypes)
print(data.index)

#print(data.dropna().describe())

#by_time = data.groupby(data.index.time).mean()
#by_time.plot()
#plt.figure()

#by_weekday = data.groupby(data.index.dayofweek).mean()
#by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
#by_weekday.plot()
#plt.figure()

ts = data["Power (kW)"]
#ts_resample = ts.resample("D").sum()
#ts_resample.plot()
#plt.figure()

ts_log = ts.rolling("7d").mean()
ts_sd = ts.rolling("7d").std()

plt.plot(ts_log)
plt.plot(ts_sd)

decomp = seasonal_decompose(ts, model='additive', freq=96)
decomp.plot()

plt.plot(ts)
plt.show()