import numpy as np
import pandas as pd
d = pd.read_csv("C:\\Users\\sivag\\github-classroom\\Saracens-High-School\\Jetlearn\\ML\\AirQuality.csv")
d.info()
X = d[["PT08.S1(CO)", "PT08.S5(O3)"]]
y = d["T"]
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size = 0.6, random_state = 10)
from sklearn.preprocessing import PolynomialFeatures
f = PolynomialFeatures(3)
pXtrain = f.fit_transform(Xtrain)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(pXtrain, ytrain)
pXtest = f.fit_transform(Xtest)
pred = lr.predict(pXtest)
from sklearn.metrics import root_mean_squared_error
error = root_mean_squared_error(ytest, pred)
print(error)