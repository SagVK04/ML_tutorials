import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

#print(diabetes.keys())
#print(diabetes.DESCR)
#diabetes_X = diabetes.data[:, np.newaxis, 2]  #makes a numpy array of the feature values only in index 2 for x-axis
diabetes_X = diabetes.data  #makes a numpy array of the feature values for all data in x-axis

#slice this dataset
diabetes_X_train = diabetes_X[:-30]  #last 30 values for training
diabetes_X_test = diabetes_X[-30:]  #first 30 values for testing        (features)

diabetes_Y_train = diabetes.target[:-30]  #y-axis for diabetes_X_train, target key value has only one column in it
diabetes_Y_test = diabetes.target[-30:]  #y axis for diabetes_X_test     (label or value)

model = linear_model.LinearRegression()  #defining model
model.fit(diabetes_X_train, diabetes_Y_train)  #fit a linear model into samples --> model training done
diabetes_Y_predicted = model.predict(diabetes_X_test)  # prediction of diabetes_Y_test through mean_squared_error
#e.g. MSE(x,y) = sum((x-y)^2)/ no. of samples

print("Mean squared error is: ", mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))
print("Weights: ",model.coef_)
print("Intercept: ",model.intercept_)

#Mean squared error is:  1826.4841712795044
#Weights:  [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067
#  458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ]
#Intercept:  153.05824267739402
#plotting is not possible as multiple independent variables are there

#plt.scatter(diabetes_X_test,diabetes_Y_test)  #original graph but scattered
#plt.plot(diabetes_X_test,diabetes_Y_predicted)  #predicted line
#plt.show()
