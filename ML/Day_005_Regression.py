# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''Simple Linear Regression
# Importing the dataset
sales = pd.read_csv("./ML/datasets/kc_house_data.csv")
X = sales.iloc[:, [5]].values
Y = sales.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting Simple Linear Regression Model to the train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(x_train, y_train)

# The coefficients, mean squared error
print( 'Benchmark Solution:' )
print( 'Intercept:', regressor.intercept_)
print( 'Slope    :', regressor.coef_[0])
print( "MSE      : %.2f" % np.mean((regressor.predict(x_test) - y_test) ** 2) )

plt.figure(figsize=(6,6), dpi=100)
plt.scatter( x_test, y_test,  color='blue', label="true" )
plt.plot( x_test, regressor.predict(x_test), color='green', label="predict" )
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.legend(loc='upper left')
plt.show()
'''

# Generative polynomial features
def polynomial_features( data, deg=2 ):
    data1 = data.copy()
    
    if isinstance(data1, pd.DataFrame):
        column_label = data1.columns.values.tolist()[0]
        data1 = data1.rename( columns={column_label: 'power_1'} )
    else:
        data1 = pd.DataFrame( np.array(data1), columns=['power_1'] )
    
    # in our example the type of data_copy['power_1'] is int64, and
    # negative number will show up when the degree is high, hence we
    # change its type to float64 to handle large number
    data1['power_1'] = data1['power_1'].astype('float64')
    
    for i in range(1,deg):
        data1['power_'+str(i+1)] = data1['power_'+str(i)]*data1['power_1']
    
    return data1

# plot
def plot_data(data): 
    plt.figure(figsize=(4,4), dpi=100)
    plt.plot( data['power_1'], data['price'], 'b.', label = 'data' )
    plt.xlabel('sqft_living')
    plt.ylabel('price')

# polynomial_regression
def polynomial_regression( data, deg=2 ):
    from sklearn.linear_model import LinearRegression
    input_value = data.iloc[:,0:deg].values
    output_value = data['price'].values
    
    #from sklearn.preprocessing import StandardScaler
    #sc = StandardScaler()
    #input_value = sc.fit_transform(input_value)

    model = LinearRegression()
    model = model.fit( input_value, output_value )
    return model

# plot_predictions
def plot_predictions( data, model ):
    deg = len(model.coef_)
    
    x_pred = polynomial_features(data['power_1'], deg )
    y_pred = model.predict( x_pred.values )
    
    # plot predictions

    #plt.scatter(data.iloc[:,0:deg].values, data['price'].values,color='blue', label="true")
    plt.scatter( x_pred['power_1'], data['price'],  color='blue', label="data" )
    plt.plot(x_pred['power_1'], y_pred, 'g-', label='degree ' + str(deg) + ' fit')
    plt.legend(loc='upper left')
    plt.show()

# print_coefficients
def print_coefficients( model ):        
    deg = len(model.coef_)
    # Get learned parameters as a list
    w = list( np.hstack((model.intercept_,model.coef_)) )

    # Numpy has a nifty function to print out polynomials in a pretty way
    print( 'Learned polynomial for degree ' + str(deg) + ':\n' )
    w.reverse()
    print( np.poly1d(w) )

# polynomial_output
def polynomial_output( set, deg=15 ):
    data = polynomial_features( set['sqft_living'], deg )
    data['price'] = set['price'].values
    model = polynomial_regression( data, deg )
    print_coefficients( model )
    plot_predictions( data, model )


# Importing the dataset and sort
sales = pd.read_csv("./ML/datasets/kc_house_data.csv")
sales = sales.sort_values( ['sqft_living','price'] )

# polynomial_output(sales, deg=1)
# polynomial_output(sales, deg=2)
# polynomial_output(sales, deg=3)
polynomial_output(sales, deg=15)
