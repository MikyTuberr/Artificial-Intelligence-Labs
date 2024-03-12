import numpy as np

class LinearReg:
    """Linear Regression"""

    @staticmethod
    def MSE(prediction, x):
        """Mean Squared Error: SUM (from i=1 to n) (actual_output - predicted_output) ** 2"""
        difference = prediction - x
        return np.mean(difference**2)

    @staticmethod
    def CFS(x, y):
        """Closed-Form Solution: theta_0 + theta_1 * x"""
        X = np.c_[np.ones_like(x), x]
        return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

    @staticmethod
    def ZS(x, my, sigma):
        """Z-score Standarization"""
        return (x - my) / sigma

    @staticmethod
    def DZS(x, my, sigma):
        """Dstandarization of Z-score Standarization"""
        return x * sigma + my

    #@staticmethod
    #def ST(theta, x_my, x_sigma, y_my, y_sigma):
        #"""Theta Scaling"""
        #theta = theta.copy()
        #theta[1] = theta[1] * y_sigma / x_sigma
        #theta[0] = y_my - theta[1] * x_my
        #return theta.reshape(-1)

    @staticmethod
    def BGD(num_iterations, learning_rate, X, y, theta):
        """Batch Gradient Descent"""
        for i in range(num_iterations):
            gradient = (2 / y.shape[0]) * X.T.dot(X.dot(theta) - y.reshape(-1, 1))
            theta -= learning_rate * gradient
        return theta