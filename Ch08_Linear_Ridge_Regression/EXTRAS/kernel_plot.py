# coding: utf-8
# Author: shelley
# 2020/5/14,15:26

import numpy as np
import matplotlib.pyplot as plt

k = 0.5
x = np.arange(0.0, 1.0, 0.01)
y = np.exp((x-1)/(-2*k**2))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y)
ax.set_ylabel('kernel')
ax.set_xlabel('x')
plt.show()