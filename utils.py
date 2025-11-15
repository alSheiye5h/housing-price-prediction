import matplotlib.pyplot as plt
import numpy as np


def load_data(path):
    data = np.loadtxt(path, delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    return X, y

def plot_data(X, y, pos_label="Positive", neg_label="Negative"):
    """
    Plots X and y into a 2D scatter plot.
    X : array of shape (m,2)
    y : array of shape (m,)
    """
    # Find positive and negative examples
    pos = y == 1
    neg = y == 0

    plt.scatter(X[pos, 0], X[pos, 1], c='b', marker='+', label=pos_label)
    plt.scatter(X[neg, 0], X[neg, 1], c='r', marker='o', label=neg_label)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
