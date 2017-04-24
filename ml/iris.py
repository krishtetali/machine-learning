import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the boston dataset
boston = datasets.load_boston()


# Use only one feature
boston_X = boston.data[:, np.newaxis, 2]

# Split the data into training/testing sets
boston_X_train = boston_X[:-20]
boston_X_test = boston_X[-20:]

# Split the targets into training/testing sets
boston_y_train = boston.target[:-20]
boston_y_test = boston.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(boston_X_train, boston_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(boston_X_test) - boston_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(boston_X_test, boston_y_test))

# Plot outputs
plt.scatter(boston_X_test, boston_y_test,  color='black')
plt.plot(boston_X_test, regr.predict(boston_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

  #out put :Mean squared error: 18.56
  #Variance score: 0.21

