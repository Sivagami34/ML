import pandas as pd
df = pd.read_csv("C:\\Users\\sivag\\github-classroom\\Saracens-High-School\\Jetlearn\\ML\\titanic.csv")
df.info()
X = df[["Pclass", "Sex", "Age", "Siblings/Spouses Aboard", "Parents/Children Aboard"]]
y = df["Survived"]
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
X["Sex"] = l.fit_transform(X["Sex"])
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size = 0.8, random_state = 42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(Xtrain, ytrain)
pred = lr.predict(Xtest)
print(pred)
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(ytest, pred)
print(cm)
print(classification_report(ytest, pred))
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(cm, annot = True)
plt.show()