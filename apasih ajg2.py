import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

path_iris = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
input_data = np.loadtxt(path_iris, delimiter=',', usecols=[0,1,2,3])
X, y = input_data[:, :-1], input_data[:, -1]
num_training = 10
training_samples = int(0.6 * len(X))
testing_samples = len(X) - num_training

X_train, y_train = X[:training_samples], y[:training_samples]

X_test, y_test = X[training_samples:], y[training_samples:]

reg_linear= linear_model.LinearRegression()
reg_linear.fit(X_train, y_train)
y_test_pred = reg_linear.predict(X_test)
# y_test_pred
# print(X_test, y_test)
# X_test
# y_test
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_test_pred, color='black', linewidth=2)
plt.xticks(())
plt.yticks(())
plt.show()