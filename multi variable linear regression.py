import pandas as pd
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
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(Xtrain, ytrain)
ptt = lr.predict(Xtest)
ptn = lr.predict(Xtrain)
from sklearn.metrics import root_mean_squared_error
etn = root_mean_squared_error(ytrain, ptn)
ett = root_mean_squared_error(ytest, ptt)
print(etn)
print(ett)
print(X)
print(lr.predict([[7.39, 16.2]]))