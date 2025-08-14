# Data Processing
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('./ML/datasets/studentscores.csv')
X = dataset.iloc[:,:1].values
Y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/4, random_state=0)

# Fitting Simple Linear Regression Model to the train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# Predicting the Result
Y_pred = regressor.predict(X_test)

# Visualising the Training Results
plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color='blue')
#plt.show()

# Visualsing the Test Results
plt.scatter(X_test, Y_test, color="red")
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.show()
