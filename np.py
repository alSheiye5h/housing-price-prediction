import numpy as np



# a = np.array([[1, 2, 3], [5, 6, 4], [1, 2, 5]])
# print(a.dtype)

# b = np.zeros(5)
# print(b)

# c = np.ones(6)
# print(c)

# d = np.empty(5)
# print(d)

# e = np.arange(6)
# print(e)

# f = np.linspace(0, 10, num=4)
# print(f)

# g = np.ones(2, dtype=np.int64)
# print(g)

a = np.array([5, 6, 4, 8, 9, 2])
b = a.reshape(3, 2)
print(b) 



print(np.reshape(a, (1, 6), order='C'))