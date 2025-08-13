#Step 1: Importing the libraries
import numpy as np
import pandas as pd

#Step 2: Importing the dataset    
dataset = pd.read_csv('./ML/datasets/Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,3].values
print('Step2: Importing the dataset')
print("X: ")
print(X)
print("Y: ")
print(Y)

#Step 3: Handing the missing data
from  sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") 
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("---------------------")
print("Step 3: Handling the missing data")
print("X: ")
print(X)

#Step 4:Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("", OneHotEncoder(), [0])],remainder='passthrough')
X = ct.fit_transform(X)
laberencoder_Y = LabelEncoder()
Y = laberencoder_Y.fit_transform(Y)
print("---------------------")
print("Step 4: Encoding categorical data")
print("X")
print(X)
print("Y")
print(Y)

#Step 5:Spliting the datasets into training sets and Test Sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("---------------------")
print("Step 5: Splitting the datasets into training sets and Test sets")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)

#Step 6:Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print("---------------------")
print("Step 6: Feature Scaling")
print("X_train")
print(X_train)
print("X_test")
print(X_test) 