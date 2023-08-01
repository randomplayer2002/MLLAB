import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
print ("Features: ", iris.feature_names)
print ("Labels: ", iris.target_names)
X=pd.DataFrame(iris['data'])

#print(X.head())
#print(iris.data.shape)
#y=print (iris.target)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.30,random_state=109)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(y_pred)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
