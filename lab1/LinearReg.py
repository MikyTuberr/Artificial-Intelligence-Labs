import numpy as np

class LinearReg:
    """Linear Regression"""

    @staticmethod
    def MSE(prediction: np.ndarray, x: np.ndarray) -> float:
        """Mean Squared Error: SUM (from i=1 to n) (actual_output - predicted_output) ** 2"""
        difference = prediction - x
        return np.mean(difference**2)

    @staticmethod
    def CFS(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Closed-Form Solution: theta_0 + theta_1 * x"""
        X = np.c_[np.ones_like(x), x]
        return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

    @staticmethod
    def ZS(x: np.ndarray, my: float, sigma: float) -> np.ndarray:
        """Z-score Standardization"""
        return (x - my) / sigma

    @staticmethod
    def DZS(x: np.ndarray, my: float, sigma: float) -> np.ndarray:
        """De-standardization of Z-score Standardization"""
        return x * sigma + my

    #@staticmethod
    #def ST(theta, x_my, x_sigma, y_my, y_sigma):
        #"""Theta Scaling"""
        #theta = theta.copy()
        #theta[1] = theta[1] * y_sigma / x_sigma
        #theta[0] = y_my - theta[1] * x_my
        #return theta.reshape(-1)

    @staticmethod
    def BGD(num_iterations: int, learning_rate: float, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Batch Gradient Descent"""
        for i in range(num_iterations):
            gradient = (2 / y.shape[0]) * X.T.dot(X.dot(theta) - y.reshape(-1, 1))
            theta -= learning_rate * gradient
        return theta
