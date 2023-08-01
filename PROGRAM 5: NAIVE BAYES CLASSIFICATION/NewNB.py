import numpy as np
import pandas as pd
from sklearn import metrics
#Import dataset 
from sklearn import datasets
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Load dataset
iris = datasets.load_iris()
print ("Features: ", iris.feature_names)
print ("Labels: ", iris.target_names)
X=pd.DataFrame(iris['data'])

#print(X.head())
#print(iris.data.shape)
#y=print (iris.target)

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.30,random_state=109)

#Create a Gaussian Classifier

gnb = GaussianNB()

#Train the model using the training sets

gnb.fit(X_train, y_train)

#Predict the response for test dataset

y_pred = gnb.predict(X_test)

print(y_pred)
# Model Accuracy

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
