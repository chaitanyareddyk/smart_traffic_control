import pandas as pd
import matplotlib.pyplot as plt
import sys


d = pd.DataFrame.from_csv(sys.argv[1], index_col=None)
d['2 seconds'] = (d['time']/(int(2)*100)).astype(int)
d = d.groupby('2 seconds').sum()
d = d.drop(['time'], axis=1)
d.plot()
plt.show()