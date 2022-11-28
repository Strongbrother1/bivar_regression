import numpy as np
import pandas as pd



df = pd.read_csv("project.csv")

fit = np.polyfit(df['X'], df['Y'], 2)
equation = np.poly1d(fit)
print ("The fit coefficients are a = {0:.4f}, b = {1:.4f} c = {2:.4f}".format(*fit))
print (equation)