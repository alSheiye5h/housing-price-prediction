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





# import numpy as np
# import os

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

# a = np.arange(10).reshape(2, -1)
# b = np.repeat(1, 10).reshape(2, -1)
# print(a)
# print(b)

# c = np.vstack([a, b])
# print(c)

# d = np.hstack([a, b])
# print(d)

# a = np.arange(3)
# b = np.repeat(a, 3)
# c = np.tile(a, 3)

# d = np.r_[b, c]

# print(d)

# a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# c = np.intersect1d(a, b)
# print(c)

# d = np.setdiff1d(a, b)
# print(d)

# a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])

# c = np.where(a == b)
# print(c)

# d = a[(a >= 5) & (a <= 10)]
# print(d)

# def maxx(x, y):
#     if x > y:
#         return x
#     else :
#         return y


# def pair_max(x, y):
#     maximum = [maxx(a, b) for a,b in map(lambda a,b:(a,b),x,y)]

# import numpy as np

# my_list = np.array([1, 5, 6, 8])
# my_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# matrix = np.array(my_matrix)
# matrix1 = np.arange(1, 10).reshape(3, 3)
# print(matrix == matrix1)

# matr = np.linspace(0, 10, 4)

# identity matrix
# identity = np.eye(4)
# print(matr)

# rand = np.random.rand(4)
# print(rand)

# rand1 = np.random.rand(5, 5)

# randarr = np.random.randint(0, 50, 10)
# print(randarr)

# print(randarr.argmax())
# print(randarr.argmin())

# arr2d = np.zeros((10, 10))
# arrlength = arr2d.shape[1]
# print(arr2d.shape[0])
# print(arrlength)

# for i in range(arrlength):
#     arr2d[i] = i

# print(arr2d)












































































