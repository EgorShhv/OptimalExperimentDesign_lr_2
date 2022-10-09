import numpy as np

f = np.matrix([[1], [20], [3], [4], [5], [6]])

# M = np.matrix([[1, 2, 3, 4, 5, 6],
#               [2, 1, 2, 3, 4 ,5],
#               [3, 2, 1, 2, 3, 4],
#               [4, 3, 2, 1, 2, 3],
#               [5, 4, 3, 2, 1, 2],
#               [6, 5, 4, 3, 2, 1]])

M = f * f.T

n = 25

grad = np.zeros(n)

for i in range(0, n):
    grad[i] = np.trace(M.T @ f * np.transpose(f))

grad2 = np.zeros(n)

for i in range(0, n):
    grad2[i] = f.T @ M.T @ f

for i in range(n):
    print("{}\t{}".format(grad[i], grad2[i]))
