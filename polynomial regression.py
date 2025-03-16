import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
X = np.arange(1, 21)
y = X**2 + 3
plt.scatter(X, y)
plt.show()
d = pd.read_csv("C:\\Users\\sivag\\github-classroom\\Saracens-High-School\\Jetlearn\\ML\\HousingData.csv")
d.info()
print(d.isnull().sum())
d.dropna(inplace = True)
print(d.isnull().sum())
#extract features and target
X = d[["RM", "LSTAT"]]
y = d["MEDV"]
#train test split
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size = 0.8, random_state = 14)
from sklearn.preprocessing import PolynomialFeatures
f = PolynomialFeatures(2)
pXtrain = f.fit_transform(Xtrain)
print(Xtrain.head(3))
print(pXtrain[0: 3])
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(pXtrain, ytrain)
print(lr.coef_)
print(lr.intercept_)
pXtest = f.fit_transform(Xtest)
pred = lr.predict(pXtest)
print(pred)
from sklearn.metrics import root_mean_squared_error
error = root_mean_squared_error(ytest, pred)
print(error)