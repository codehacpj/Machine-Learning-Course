#polynomial regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set here we don't do that as data available is very less
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2) 
X_poly = poly_reg.fit_transform(X) #making new features from X... X^2,X^3... till X^n determined by degree and it adds the column of 1's to include b0 in the equation y = b0 + b1x1 + b2(x1)^2 + b3(x1)^3 +...+ bn*(x1)^n automatically.
poly_reg.fit(X_poly,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#visualising the lin. reg. results
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')


#Visualising the pol. reg. results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='black')


#Adding the degree may help us get a better model.
poly_reg_2 = PolynomialFeatures(degree=4) 
X_poly_2 = poly_reg_2.fit_transform(X) 
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly_2,y)

#visualizing with the additional degree
plt.plot(X_grid,lin_reg_3.predict(poly_reg_2.fit_transform(X_grid)),color='orange')
plt.show()

#Predicting a new result with linear regression
print("The expected salary accon. lin. reg. is: ",lin_reg.predict(6.5))


#predicting a new result with polynomial regression
print("The expected salary accon. pol. reg. is: ",lin_reg_2.predict(poly_reg.fit_transform(6.5)))
print("The expected salary accon. tuned pol. reg. is: ",lin_reg_3.predict(poly_reg_2.fit_transform(6.5)))




