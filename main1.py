import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

data_X = np.array([[1,3],[2,1],[3,2]])  #used for both training & testing as x

data_Y = np.array([[3,4],[2,3],[4,2]])  #used for both training & testing as x

model = linear_model.LinearRegression()
model.fit(data_X,data_Y)
data_Y_P = model.predict(data_X)

print("Mean squared error: ",mean_squared_error(data_Y_P,data_Y))
print("Weight value(m): ",model.coef_)
print("Intercept value(b): ",model.intercept_)
print("Original values: ",data_Y)
print("Predicted values: ",data_Y_P)

plt.scatter(data_X,data_Y)
plt.plot(data_X,data_Y_P)
plt.show()