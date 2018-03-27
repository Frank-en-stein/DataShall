import numpy as np, matplotlib.pyplot as plot
np.seterr(divide='ignore')

x = np.arange(-10,10,.25)
y = 0
try:
    y = -1/((x+2)**2) + 4
except:
    pass
plot.plot(x,y)
plot.show()
