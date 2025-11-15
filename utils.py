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

def plot_decision_boundary(w, b, X, y):
    # Plot the data points
    pos = y == 1
    neg = y == 0
    plt.scatter(X[pos, 0], X[pos, 1], c='b', marker='o', label='Admitted')
    plt.scatter(X[neg, 0], X[neg, 1], c='r', marker='x', label='Not admitted')

    # Plot the decision boundary
    x_values = [np.min(X[:, 0])-2, np.max(X[:, 0])+2]
    # For logistic regression: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b)/w2
    y_values = -(w[0]*np.array(x_values) + b)/w[1]

    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.show()

def map_feature(x1, x2, degree=6):
    """
    Maps two input features to polynomial features up to the given degree.
    
    Args:
        x1, x2 : array-like, shape (m,)
        degree : maximum degree of polynomial features

    Returns:
        out : array, shape (m, number of features)
    """
    if x1.ndim > 0:
        x1 = x1[:, np.newaxis]
    if x2.ndim > 0:
        x2 = x2[:, np.newaxis]

    out = np.ones_like(x1)  # bias term
    for i in range(1, degree + 1):
        for j in range(i + 1):
            term = (x1 ** (i - j)) * (x2 ** j)
            out = np.hstack((out, term))
    return out