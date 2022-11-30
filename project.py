# sklearn package
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
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

print('Intercept: \n', reg_mod.intercept_)
print('Coefficients \n', reg_mod.coef_)

mean_squared_error(y_vals, y_pred, squared=False)

test_x = 87.54096453008634
test_y = -77.77999732554369
test_a = 9.26618031e-04
test_b = -1.00113557e+00
test_a2 = 1.20000059e+01
test_ab = 2.99999416e+00
test_b2 = -6.03159917e-07
equation = reg_mod.intercept_ + (test_a * test_x) + (test_b * test_y) + (test_a2 * test_x * test_x) + (test_ab * test_x * test_y) + (test_b2 * test_y * test_y)

print(equation)
# check our accuracy for each degree, the lower the error the better!
"""number_degrees = [1,2,3,4,5,6,7]
plt_mean_squared_error = []
for degree in number_degrees:

   poly_model = PolynomialFeatures(degree=degree)
  
   poly_x_values = poly_model.fit_transform(x_vals)
   poly_model.fit(poly_x_values, y_vals)
  
   regression_model = LinearRegression()
   regression_model.fit(poly_x_values, y_vals)
   y_pred = regression_model.predict(poly_x_values)
  
   plt_mean_squared_error.append(mean_squared_error(y_vals, y_pred, squared=False))"""
  
# plt.scatter(number_degrees,plt_mean_squared_error, color="green")
# plt.plot(number_degrees,plt_mean_squared_error, color="red") 


"""
1
a = 9.26618031e-04
b = -1.00113557e+00
a^2 = 1.20000059e+01
ab = 2.99999416e+00
b^2 = -6.03159917e-07


20.129297290460755,14.059520911264144,5707.43434233776

1 + 7663 x 20 + 6050 x 14 + -6.8
"""