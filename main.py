import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math


# load dataset
X_train, y_train = load_data("data.txt")

# # check data is loaded
# print("First five elements in X_train are:\n", X_train[:5])
# print("Type of X_train:",type(X_train))

# print("First five elements in y_train are:\n", y_train[:5])
# print("Type of y_train:",type(y_train))

# # exacly a 100 training example
# print ('The shape of X_train is: ' + str(X_train.shape))
# print ('The shape of y_train is: ' + str(y_train.shape))
# print ('We have m = %d training examples' % (len(y_train)))

# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
# plt.show()


# the sigmoid function : 
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    
    return g

value = 0

print (f"sigmoid({value}) = {sigmoid(value)}")

def compute_cost(X, y, w, b, *argv):

    m, n = X.shape
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    
    cost = (-1 / m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    
    return cost

m, n = X_train.shape

# Compute and display cost with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))

# Compute and display cost with non-zero w and b
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)

print('Cost at test w and b (non-zeros): {:.3f}'.format(cost))


# UNIT TESTS
compute_cost_test(compute_cost)