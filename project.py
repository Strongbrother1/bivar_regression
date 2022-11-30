from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

import pandas as pd
import numpy as np
import os

# Checks cwd for possible problems and locates the directory
# print(os.path.join(os.getcwd(), 'project.csv'))

# Read dataframe and set values
df = pd.read_csv("project.csv")

x_vals = df[['X', 'Y']].values
y_vals = df['Z'].values



## Fitting X-values

degree = 2
poly_model = PolynomialFeatures(degree=degree)

poly_x_vals = poly_model.fit_transform(x_vals)


## Fitting Y-values

poly_model.fit(poly_x_vals, y_vals)
reg_mod = LinearRegression()
reg_mod.fit(poly_x_vals, y_vals)

y_pred = reg_mod.predict(poly_x_vals)

print('Intercept: \n', reg_mod.intercept_)
print('Coefficients \n', reg_mod.coef_)

mean_squared_error(y_vals, y_pred, squared=False)

# Row number in csv to test
example_num = 100

# Prints out first row of values
print('Row values are : \n', x_vals[example_num], y_vals[example_num])

example_x = x_vals[example_num][0]
example_y = x_vals[example_num][1]
test_a = reg_mod.coef_[1]
test_b = reg_mod.coef_[2]
test_a2 = reg_mod.coef_[3]
test_ab = reg_mod.coef_[4]
test_b2 = reg_mod.coef_[5]
Actual = reg_mod.intercept_ + (test_a * example_x) + (test_b * example_y) + (test_a2 * example_x * example_x) + (test_ab * example_x * example_y) + (test_b2 * example_y * example_y)
Expected = y_vals[example_num]
print('Actual value is : ', Actual)
print('Expected value is : ', Expected)


"""
Z = a + bX + cY + dX^2 + eXY + fY^2

a = 6.955425043073774
b = 9.26618031e-04
c = -1.00113557e+00
d = 1.20000059e+01
e = 2.99999416e+00
f = -6.03159917e-07

So Equation for this specific CSV is :
Z = 6.955425043073774 + 9.26618031e-04x + -1.00113557e+00y + 1.20000059e+01x^2 + 2.99999416e+00xy + -6.03159917e-07y^2
"""