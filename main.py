import numpy as np
import math
from numpy.linalg import inv


def f(x1, x2):
    f = np.matrix([[1], [x1], [x2], [x1 * x2], [x1 * x1], [x2 * x2]])
    return f


def coef_gradient_descent(n, x, p):
    # Проекция вектора градиента (?)
    proj = np.zeros(n)
    grad = np.zeros(n)

    delta = 1e-3

    fout = open("new_out.txt", "w")
    is_solving = True
    iter = 0
    while is_solving and iter < 5e+3:
        lambdas = 1e-1
        is_solving = False
        q = n - np.count_nonzero(p)

        M = np.zeros((6, 6))
        for i in range(n):
            M += p[i] * ((f(x[i][0], x[i][1])) * np.transpose(f(x[i][0], x[i][1])))
        D = inv(M)

        # 10.801353158491526
        for i in range(n):
            grad[i] = np.transpose(f(x[i][0], x[i][1])) @ D @ (f(x[i][0], x[i][1]))

        grad /= np.linalg.norm(grad)

        # Вычисляем среднюю сумму компонент градиента точек, в которых p_j != 0
        # Это нужно для проверки условия о достаточности (6)
        avg = 0.0
        for i in range(n):
            if p[i] != 0.0:
                avg += grad[i]
        avg /= (n - q)

        # Условие необходимости (5)
        # for i in range(n-1):
        #     not_equals = False
        #     for j in range(i, n):
        #         if abs(grad[j] - grad[i]) > delta:
        #             is_solving = True
        #             not_equals = True
        #             break
        #         # else:
        #         #     p[i + 5 * j] = 0
        #     if not_equals:
        #         break
        for i in range(0, n):
            if p[i] != 0:
                if abs(grad[i] - avg) > 1e-15:
                    is_solving = True



        # # Условие необходимости (5) НЕВЕРНОЕ
        # for i in range(n):
        #     if p[i] != 0:
        #         if abs(grad[i] - avg) > delta:
        #             solution = 0

        # Условие достаточности (6)
        for j in range(n):
            proj[j] = grad[j] - avg    # Формула (7)
            if p[j] == 0:
                if grad[j] > avg:
                    is_solving = True
                    break
                else:
                    proj[j] = 0

        # for j in range(0, n):
        #     proj[j] = grad[j] - avg
        #     if p[j] == 0:
        #         if (proj[j] > 0):
        #             is_solving = True
        #         else:
        #             proj[j] = 0

        # # Вывод информации о каждой 100-й итерации
        # if (iter % 100 == 0):
        #     fout.write("iter= " + str(iter) + " ")
        #     fout.write("detM= " + str(np.linalg.det(M)) + " ")
        #     fout.write("|proj|= " + str(np.linalg.norm(proj)) + "\n")

        if is_solving:
            for i in range(n):
                if (proj[i] < 0 and lambdas > - p[i] / proj[i]):
                    lambdas = - p[i] / proj[i]

            for i in range(n):
                p[i] += lambdas * proj[i]   # Шаг 5 в методичке

        iter += 1

    fout.write("SOLUTION")
    fout.write("\niter= " + str(iter) + "\np: ")
    for i in range(n):
        fout.write(str(p[i]) + " ")
    fout.write("\ndetM= " + str(np.linalg.det(M)) + " ")
    fout.write("|proj|= " + str(np.linalg.norm(proj)) + "\n")
    print("Count iter = {}".format(iter))
    return p, D


def is_optimal(D, p):
    n = 5
    max = 0

    x1max = -1
    x2max = -1

    for i in range(n):
        for j in range(n):
            tr = np.trace((f(-1 + 0.5 * i, -1 + 0.5 * j) * np.transpose(f(-1 + 0.5 * i, -1 + 0.5 * j))) * D)
            if tr > max:
                max = tr
                x1max = -1 + 0.5 * i
                x2max = -1 + 0.5 * j

    return max, x1max, x2max


m = 5

n = m * m

p = np.ones(n)
p = p * (1.0 / n)

x = np.zeros((n, 2))

for i in range(0, m):
    for j in range(0, m):
        x[i * m + j][0] = -1 + 0.5 * i
        x[i * m + j][1] = -1 + 0.5 * j

p, D = coef_gradient_descent(n, x, p)
max, x1m, x2m = is_optimal(D, p)
print(max)
