import numpy as np
import math
from numpy.linalg import inv


def f(x1, x2):
    f = np.matrix([[1], [x1], [x2], [x1 * x2], [x1 * x1], [x2 * x2]])
    return f


def coef_gradient_descent(n, x, p):
    proj = np.zeros(n)
    grad = np.zeros(n)

    fout = open("new_out.txt", "w")
    solution = 0
    iter = 0
    while solution == 0:
        lambdas = 0.1
        solution = 1
        q = n - np.count_nonzero(p)

        M = np.matlib.zeros((6, 6))
        for i in range(0, n):
            M += p[i] * ((f(x[i][0], x[i][1])) * np.transpose(f(x[i][0], x[i][1])))
        M_1 = inv(M)

        for i in range(0, n):
            grad[i] = np.trace(M_1 * (f(x[i][0], x[i][1])) * np.transpose(f(x[i][0], x[i][1])))

        grad /= np.linalg.norm(grad)

        avg = 0.0
        for i in range(0, n):
            if p[i] != 0.0:
                avg += grad[i]
        avg /= n - q

        for i in range(0, n):
            if (p[i] != 0):
                if (abs(grad[i] - avg) > 1e-10):
                    solution = 0

        for j in range(0, n):
            proj[j] = grad[j] - avg
            if p[j] == 0:
                if (proj[j] > 0):
                    solution = 0
                else:
                    proj[j] = 0

        if (iter % 100 == 0):
            fout.write("iter= " + str(iter) + " ")
            fout.write("detM= " + str(np.linalg.det(M)) + " ")
            fout.write("|proj|= " + str(np.linalg.norm(proj)) + "\n")

        if (solution == 0):
            for i in range(0, n):
                if (proj[i] < 0 and lambdas > - p[i] / proj[i]):
                    lambdas = - p[i] / proj[i]

            for i in range(0, n):
                p[i] += lambdas * proj[i]

        iter += 1

    fout.write("SOLUTION")
    fout.write("\niter= " + str(iter) + "\np: ")
    for i in range(0, n):
        fout.write(str(p[i]) + " ")
    fout.write("\ndetM= " + str(np.linalg.det(M)) + " ")
    fout.write("|proj|= " + str(np.linalg.norm(proj)) + "\n")

    return p, M_1


def is_optimal(M_1):
    n = 201
    max = 0
    for i in range(0, n):
        for j in range(0, n):
            tr = np.trace((f(-1 + 0.01 * i, -1 + 0.01 * j) * np.transpose(f(-1 + 0.01 * i, -1 + 0.01 * j))) * M_1)
            if (tr > max):
                max = tr
                x1m = -1 + 1e-2 * i
                x2m = -1 + 1e-2 * j
    return max, x1m, x2m


m = 21
n = m * m
p = np.ones(n)
p = p * (1.0 / n)
x = np.zeros((n, 2))

for i in range(0, m):
    for j in range(0, m):
        x[i * m + j][0] = -1 + 0.1 * i
        x[i * m + j][1] = -1 + 0.1 * j

p, M_1 = coef_gradient_descent(n, x, p)
max, x1m, x2m = is_optimal(M_1)
print(max)
