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

# a = np.array([5, 6, 4, 8, 9, 2])
# b = a.reshape(3, 2)
# print(b) 

# print(np.reshape(a, (2, 3), order='C'))

# a = np.array([1, 2, 3, 4, 5, 6])
# print(a)
# print(a.shape)

# a2 = a[np.newaxis, :]
# print(a2)
# print(a2.shape) 

# a = np.array([1, 2, 3, 4, 5, 6])
# row_vector = a[np.newaxis, :]
# print(row_vector.shape)

# col_vector = a[:, np.newaxis]
# print(col_vector)
# print(col_vector.shape)

a = np.array([1, 2, 3, 4, 5, 6])
b = np.expand_dims(a, axis=1)
print(b)
print(b.shape)

























