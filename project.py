# sklearn package
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# dataframes
import pandas as pd
# computation
import numpy as np
# visualization
import matplotlib.pyplot as plt
# import os

# Checks cwd for possible problems
# print(os.path.join(os.getcwd(), 'project.csv'))
df = pd.read_csv("project.csv")

x_vals = df[['X', 'Y']].values
y_vals = df['Z'].values

print(x_vals[0], y_vals[0])


## Fitting X-values

degree = 2
poly_model = PolynomialFeatures(degree=degree)

poly_x_vals = poly_model.fit_transform(x_vals)

print(f'initial values {x_vals[0]}\nMapped to {poly_x_vals[0]}')

## Fitting Y-values

poly_model.fit(poly_x_vals, y_vals)
reg_mod = LinearRegression()
reg_mod.fit(poly_x_vals, y_vals)

y_pred = reg_mod.predict(poly_x_vals)
reg_mod.coef_

mean_squared_error(y_vals, y_pred, squared=False)


# check our accuracy for each degree, the lower the error the better!
number_degrees = [1,2,3,4,5,6,7]
plt_mean_squared_error = []
for degree in number_degrees:

   poly_model = PolynomialFeatures(degree=degree)
  
   poly_x_values = poly_model.fit_transform(x_vals)
   poly_model.fit(poly_x_values, y_vals)
  
   regression_model = LinearRegression()
   regression_model.fit(poly_x_values, y_vals)
   y_pred = regression_model.predict(poly_x_values)
  
   plt_mean_squared_error.append(mean_squared_error(y_vals, y_pred, squared=False))
  
plt.scatter(number_degrees,plt_mean_squared_error, color="green")
plt.plot(number_degrees,plt_mean_squared_error, color="red") 

"""fit = np.polyfit(df['X'], df['Y'], 2)
equation = np.poly1d(fit)
print ("The fit coefficients are a = {0:.4f}, b = {1:.4f} c = {2:.4f}".format(*fit))
print (equation)"""