#Multiple regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding the categorical variables.

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:,1:] # the lin reg library is taking care of the dummy variable trap still for the sake of understanding we show it over here

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#linear regression lib will take care of the feature scaling too.
# Feature Scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting the multiple lin. regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

#regressor.score(X_test,y_test) # the model may have statistically insignificant variables
#inorder to build the optimal model usig backward elimination
#making the feature set consistent with y = b0*x0 + b1*x1 + b2*x2 + b3*x3 +...+ bn*xn
#to do so we: y = b0 + b1*x1 + b2*x2 + b3*x3 +...+ bn*xn we need to add the x0---> vector of 1's
 
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values=X,axis = 1)

#backward elimination t find the optimal feature set
X_opt = X[:,[0,1,2,3,4,5]] #initial optimal is with all the features.
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(regressor_OLS.summary())
#by looking at the summary we find that the feature X2 has the most p-value and is way above 5% ie SL = 0.05
#so we remove the X2 which is at index 2 in X. ie new X_opt = X[:,[0,1,3,4,5]]

X_opt = X[:,[0,1,3,4,5]] 
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(regressor_OLS.summary())

#now X1 has the most p-value greater than 0.05
X_opt = X[:,[0,3,4,5]] 
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(regressor_OLS.summary())

#again X2 has the most and greater than 0.05i.e feature 4 has to be removed.
X_opt = X[:,[0,3,5]] 
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(regressor_OLS.summary())

#still x2 has 0.06 p-value, strictly following the slide we remove the feature 5 
# but to be sure we look at more powerful metrics like r squared and adjusted r squared.
X_opt = X[:,[0,3]] 
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(regressor_OLS.summary())











