import numpy as np
import matplotlib.pyplot as plt
from LinearReg import LinearReg as lr
from data import get_data, inspect_data, split_data

def plot_regression_line(x, y, theta):
    x_values = np.linspace(min(x), max(x), 100)
    theta = np.squeeze(theta)
    y_values = float(theta[0]) + float(theta[1]) * x_values
    plt.plot(x_values, y_values, "red")
    plt.scatter(x, y)
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.show()

def standardize_data(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return lr.ZS(data, mean, std_dev), mean, std_dev

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Get columns
y_train = train_data['MPG'].to_numpy().reshape(-1, 1)
x_train = train_data['Weight'].to_numpy().reshape(-1, 1)
y_test = test_data['MPG'].to_numpy().reshape(-1, 1)
x_test = test_data['Weight'].to_numpy().reshape(-1, 1)

# Calculate closed-form solution
theta_best = lr.CFS(x_train, y_train)
print("\nClosed-form solution theta:", theta_best.flatten())

# Calculate error using closed-form solution
prediction = theta_best[0] + theta_best[1] * x_test
print("MSE (closed-form):", lr.MSE(prediction, y_test))

plot_regression_line(x_test, y_test, theta_best)

# Standardize data
x_train, x_mean, x_std = standardize_data(x_train)
y_train, y_mean, y_std = standardize_data(y_train)
x_test = lr.ZS(x_test, x_mean, x_std)

# Calculate theta using Batch Gradient Descent
X_train = np.c_[np.ones_like(x_train), x_train]
theta_best = np.random.randn(2, 1)
learning_rate = 0.001
num_iterations = 16465

theta_best = lr.BGD(num_iterations, learning_rate, X_train, y_train, theta_best)
print("Batch Gradient Descent theta:", theta_best.flatten())

# Calculate error using Batch Gradient Descent
y_test_predicted = theta_best[0] + theta_best[1] * x_test
y_test_predicted = lr.DZS(y_test_predicted, y_mean, y_std)
print("MSE (Batch Gradient Descent):", lr.MSE(y_test_predicted, y_test))

y_test = lr.ZS(y_test, y_mean, y_std)
plot_regression_line(x_test, y_test, theta_best)
