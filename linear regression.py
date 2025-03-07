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

#evaluating the model
#RMSE - Root Mean Squared Error sqrt( sum( (p – yi)^2 )/n )
error = np.sqrt(np.mean(((predy - y) ** 2)))
print(error)
#using library
from sklearn.linear_model import LinearRegression
x = x.reshape(-1, 1)
lr = LinearRegression()
lr.fit(x, y)
print("Slope: ", lr.coef_)
print("Intercept: ", lr.intercept_)
predy = lr.predict(x)
print(predy)
plt.scatter(x, y)
plt.plot(x, predy)
plt.show()

from sklearn.metrics import root_mean_squared_error
errory = root_mean_squared_error(y, predy)
print(errory)
