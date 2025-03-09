import pandas as pd
import numpy as np
d = pd.read_csv("C:\\Users\\sivag\\github-classroom\\Saracens-High-School\\Jetlearn\ML\\Salary.csv")
d.info()
#separating feature and target
X = d["YearsExperience"]
y = d["Salary"]

#linear regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
X = np.array(X)
X = X.reshape(-1, 1)
lr.fit(X, y)
pred = lr.predict(X)

#evaluate model
from sklearn.metrics import root_mean_squared_error
ey = root_mean_squared_error(y, pred)
print(ey)

#testing
s = int(input("Enter years of experience: "))
p = lr.predict([[s]])
print(p)