import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\k\Downloads\HeightVsWeight.csv')
ind = dataset.iloc[:,1:-1].values
dep = dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression

linearRegression = LinearRegression()
linearRegression.fit(ind, dep)

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=4)
ind_poly = poly_features.fit_transform(ind)
polyLinearRegression = LinearRegression()
polyLinearRegression.fit(ind_poly, dep)

plt.scatter(ind, dep, color='red')
plt.plot(ind, linearRegression.predict(ind), color='blue')
plt.title("Regression Linear Simple")
plt.xlabel("Nivel")
ply.ylabel("Salario")
plt.show()

plt.scatter(ind, dep="red")
plt.plot(ind, polyLinearRegression.predict(ind_poly), color='blue')
plt.title('Regression Linear Polinomial')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

print(linearRegression.predict([15.7]))

print(linearRegression.predict([[0]]))

print("y = "+linearRegression.coef_[0]+"x +"+linearRegression.intercept)