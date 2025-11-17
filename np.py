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

# a = np.array([1, 2, 3, 4, 5, 6])
# b = np.expand_dims(a, axis=1)
# print(b)
# print(b.shape)

# c = np.expand_dims(a, axis=0)
# print(c)
# print(c.shape)

# data = np.array([1, 2, 3])
# print(data[1])
# print(data[0:2])
# print(data[1:])
# print(data[-2:])

# a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# print(a[a < 6])

# five_up = (a >= 5)
# print(a[five_up])

# a1 = np.array([[1, 1], [2, 2]])
# a2 = np.array([[3, 3], [4, 4]])

# print(np.vstack((a1, a2)))
# print(np.hstack((a1, a2)))

# x = np.arange(1, 25).reshape(2, 12)
# print(x)

# print(np.hsplit(x, 3))

# data = np.array([1, 2])
# other = np.array([5, 6])

# print(data + other)

# import pandas as pd
# import numpy as np
# np.random.seed(seed=1234)

# url = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/titanic.csv"
# df = pd.read_csv(url, header=0)

# df.head(3)  





import numpy as np
import os

# for dirname, _, filenames in os.walk('../input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = 'all'

# print(np.__version__)

# a = np.arange(10)
# print(a)

# b = np.full((3, 3), True, dtype=bool)
# c = np.full((9), True, dtype=bool).reshape(3, 3)
# d = np.ones((3, 3), dtype=bool)
# e = np.ones((9), dtype=bool)
# print(b)
# print(c)
# print(d)
# print(e)

# arr = np.arange(10)

# two_arr = arr[arr % 2 == 1] 
# three_arr = arr
# three_arr[three_arr % 2 == 1] = -1

# print(two_arr)
# print(three_arr)


# arr = np.arange(10)

# out = arr.reshape(2, 5)

# print(out)

































































