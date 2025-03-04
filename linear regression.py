import numpy as np
import matplotlib.pyplot as plt
x = np.arange(1, 11)
y = [23, 26, 27, 34, 38, 39, 45, 47, 48, 50]
y = np.array(y)
plt.scatter(x,y)
plt.show()

#find line of best fit
#m = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi – mean(x))^2)
#c = mean(y) – m * mean(x)
meanx = x.mean()
meany =y.mean()
m = np.sum((x - meanx) * (y - meany)) / np.sum((x - meanx)**2)
c = meany - (m * meanx)
print("slope ", m)
print("intercept", c)

#prediction
predy = m*x + c
plt.scatter(x,y)
plt.plot(x, predy)
plt.show()