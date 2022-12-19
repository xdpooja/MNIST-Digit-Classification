import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

testdf = pd.read_csv(r'D:/python/ml/mnist_test.csv')
traindf = pd.read_csv(r'D:/python/ml/mnist_train.csv')

#preprocessing
x_train = traindf.drop('label',axis=1).iloc[0:1000,0:1000]
y_train = traindf['label'].iloc[0:1000]
x_test = testdf.drop('label',axis=1).iloc[0:1000,0:1000]
y_test = testdf['label'].iloc[0:1000]
print(x_train.shape)
print(x_test.shape)
y_test

#test image
plt.figure(figsize=(3,3))
random_num= int(input("enter random number: "))
image = x_train.iloc[random_num].to_numpy().reshape(28,28)
plt.imshow(image, cmap=matplotlib.cm.binary)
print(y_train[random_num])

#SVM model
from sklearn.svm import SVC
svml = SVC(kernel="linear")
svml.fit(x_train, y_train)
pred_svml = svml.predict(x_test)
print("Mean squared error svm(linear): ",mean_squared_error(y_test, pred_svml))
print("Accuracy svm(linear): ",accuracy_score(y_test, pred_svml))
svmp = SVC(kernel="poly", degree = 2)
svmp.fit(x_train, y_train)
pred_svmp = svmp.predict(x_test)
print("Mean squared error svm(polynomial): ",mean_squared_error(y_test, pred_svmp))
print("Accuracy svm(polynomial): ",accuracy_score(y_test, pred_svmp))
svm_rbf = SVC(kernel="rbf")
svm_rbf.fit(x_train, y_train)
pred_svm_rbf = svm_rbf.predict(x_test)
print("Mean squared error svm(rbf): ",mean_squared_error(y_test, pred_svm_rbf))
print("Accuracy svm(rbf): ",accuracy_score(y_test, pred_svm_rbf))

#logistic regression
from sklearn.linear_model import LogisticRegression  
lr = LogisticRegression()
lr.fit(x_train, y_train)
pred_lr = lr.predict(x_test)
print("Mean squared error of Logistic Regression :",mean_squared_error(y_test, pred_lr))

#Naive Bayes
from sklearn.naive_bayes import GaussianNB 
nb = GaussianNB()
nb.fit(x_train, y_train)
pred_nb = nb.predict(x_test)
print("Mean squared error of Naive Bayes :",mean_squared_error(y_test, pred_nb))

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)
print("Mean squared error of KNN :",mean_squared_error(y_test, pred_knn))
