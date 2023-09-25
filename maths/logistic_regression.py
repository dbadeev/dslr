"""
Module with logistic regression class and functions
"""
import numpy as np
import pandas as pd

from messages import Messages


def sigmoid(w: np.ndarray, x: pd.DataFrame) -> np.ndarray:  # pylint: disable=C0103
    """
    Function for calculating predicted value probabilities

    :param w: weights for features
    :param x: data
    :return: prediction
    """
    return np.divide(1, np.add(1, (np.exp(-x.dot(w)).astype(np.float))))


def log_deriv(x_val: np.array, f_x_val: np.array, weights: np.ndarray) -> list:
    """
    Gradient function

    :param x_val: features
    :param f_x_val: f(x)
    :param weights: weights for features
    :return: gradient list
    """
    return x_val.T.dot(np.subtract(sigmoid(weights, x_val), f_x_val)) \
           / x_val.shape[0]


def get_mini_batch(x_val, f_x_val, b):  # pylint: disable=C0103
    """
    Yields

    :param x_val:
    :param f_x_val:
    :param b:
    :return:
    """
    last = 0
    counter = 0
    while True:
        for i in range(b, x_val.shape[0], b):
            yield x_val[last:i], f_x_val[last:i], counter * x_val.shape[0] + i
            last = i
        counter += 1


def shuffle(x_val: pd.DataFrame, f_x_val: pd.Series) -> (np.ndarray, np.ndarray):
    """
    Used for sgd method to shuffle data

    :param x_val: features data
    :param f_x_val: f(x) data
    :return: shuffled x_set and y_set
    """
    shuffle_ = np.column_stack((x_val, f_x_val))
    np.random.shuffle(shuffle_)
    shuffle_ = shuffle_.transpose()
    return (shuffle_[:-1]).transpose(), (shuffle_[-1:]).transpose()


class LogisticRegression:
    """
    Logistic regression class
    """
    def __init__(self, grad='batch', weights=None, alpha=0.0001,
                 n_cycle=1000000, b=8):  # pylint: disable=C0103
        self.grad = grad
        self.weights = weights
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.b = b  # pylint: disable=C0103  # b == mini-batch size,  2 =< b =< 32, here 18

    def batch(self, x_val: pd.DataFrame, f_x_val: pd.Series, debug: bool):
        """
        Default gradient descend method. Each gradient descend step uses all
        data(rows), one n_cycle equals one step

        :param x_val: features data
        :param f_x_val: f(x) data
        :param debug: True / False param for debug
        """
        for _ in range(0, self.n_cycle):
            gradient = log_deriv(x_val, f_x_val, self.weights)
            self.weights = np.subtract(self.weights, (self.alpha * gradient))
            if debug is True:
                if ((_ * 500) / self.n_cycle) % 10 == 0:
                    prediction = sigmoid(self.weights, x_val)
                    loss_ = self.loss(prediction, f_x_val)
                    bin_predict = (np.asarray(prediction) > 0.5).astype(np.int_)
                    accuracy = (bin_predict == f_x_val).sum() / f_x_val.shape[0]
                    Messages(f'Iteration {_}: Loss: '
                             f'{str(np.hstack(loss_))[1:-1]} - Current '
                             f'Accuracy: {accuracy}').info_()
        if debug is True:
            Messages(f'Weights: {str(np.hstack(self.weights))[1:-1]}').info_()

    def sgd(self, x_val: pd.DataFrame, f_x_val: pd.Series, debug: bool):
        """
        Stochastic gradient descend method. Each gradient descend step uses one
        data(row), one n_cycle equals one step. Good for huge datasets

        :param x_val: features data
        :param f_x_val: f(x) data
        :param debug: True / False param for debug
        """
        x_set, y_set = shuffle(x_val, f_x_val)
        for _ in range(0, x_set.shape[0]):
            if _ > self.n_cycle:
                break
            gradient = log_deriv(np.array([x_set[_]]), np.array([y_set[_]]),
                                 self.weights)
            self.weights = np.subtract(self.weights, (self.alpha * gradient))
            if debug is True:
                if ((_ * 100) / x_set.shape[0]) % 4 == 0:
                    prediction = sigmoid(self.weights, x_val)
                    loss_ = self.loss(prediction, f_x_val)
                    bin_predict = (np.asarray(prediction) > 0.5).astype(np.int_)
                    accuracy = (bin_predict == f_x_val).sum() / f_x_val.shape[0]
                    Messages(f'Iteration {_}: Loss: '
                             f'{str(np.hstack(loss_))[1:-1]} - Current '
                             f'Accuracy: {accuracy}').info_()
        if debug is True:
            Messages(f'Weights: {str(np.hstack(self.weights))[1:-1]}').info_()

    def mini_batch(self, x_val, f_x_val, debug):
        """
        Mini-batch gradient descend method. Each gradient descend step uses b
        data(rows), one n_cycle equals one step. Compromise between batch and
        sgd. Good for huge datasets

        :param x_val: features data
        :param f_x_val: f(x) data
        :param debug: True / False param for debug
        """
        for x, y, _ in get_mini_batch(x_val, f_x_val, self.b):  # pylint: disable=C0103
            if _ > x_val.shape[0]:
                break
            gradient: list = log_deriv(x, y, self.weights)
            self.weights = np.subtract(self.weights, (self.alpha * gradient))
            if debug is True:
                if (_ / self.b) % 3 == 0 or _ == self.b:
                    prediction = sigmoid(self.weights, x_val)
                    loss_ = self.loss(prediction, f_x_val)
                    # print(f'loss = {loss_}')
                    bin_predict = (np.asarray(prediction) > 0.5).astype(np.int_)
                    accuracy = (bin_predict == f_x_val).sum() / f_x_val.shape[0]
                    Messages(f'Iteration {int(_ / self.b)}: Loss: '
                             f'{str(np.hstack(loss_))[1:-1]} - Current '
                             f'Accuracy: {accuracy}').info_()
        if debug is True:
            Messages(f'Weights: {str(np.hstack(self.weights))[1:-1]}').info_()

    def fit(self, val_houses: pd.DataFrame, debug: bool):
        """
        Fit uses gradient descend, but different gradient descend methods can
        be used.

        :param val_houses: hogwarts houses data
        :param debug: bool value for debug mode
        """
        f_x_val = val_houses.iloc[:, 0].values
        f_x_val.shape = (len(f_x_val), 1)
        x_val = val_houses.iloc[:, 1:].values
        if self.grad == "batch":
            self.batch(x_val, f_x_val, debug)
        elif self.grad == "mini_batch":
            self.mini_batch(x_val, f_x_val, debug)
        elif self.grad == "sgd":
            self.sgd(x_val, f_x_val, debug)

    @staticmethod
    def loss(predicted_values: np.array, expected_values: np.array):
        """
        Loss functions

        :param predicted_values: predicted data
        :param expected_values: actual data
        :return: loss
        """
        one = np.ones(predicted_values.shape)
        loss = expected_values.transpose().dot((np.log(predicted_values))) + (
            np.subtract(one, expected_values).transpose().dot(
                (np.log(np.subtract(one, predicted_values)))))
        loss = (loss / predicted_values.shape[0] * -1)
        return loss
