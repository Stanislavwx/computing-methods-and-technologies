"""
Лабораторна робота №7
Тема: Ітераційні методи розв’язку систем лінійних алгебраїчних рівнянь
      (простої ітерації, Якобі, Зейделя)
Дисципліна: Методи та технології обчислень
Виконав: студент 3 курсу групи ФеП-32 Чепара Станіслав
"""

import os
import numpy as np


def generate_system(n=100, x_val=2.5, matrix_file="matrix_A.txt", vector_file="vector_B.txt", seed=26):
    """
    Генерує випадкову матрицю A розмірності n x n зі строгим діагональним
    переважанням та обчислює вектор вільних членів B за контрольним розв'язком x_i = x_val.
    """
    np.random.seed(seed)
    # Початкова генерація позадіагональних елементів
    A = np.random.uniform(-10.0, 10.0, (n, n))

    # Забезпечення строгого діагонального переважання: |a_ii| > sum_{j != i} |a_ij|
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        A[i, i] = row_sum + np.random.uniform(5.0, 15.0)

    # Контрольний вектор точного розв'язку
    x_exact = np.full(n, x_val, dtype=np.float64)

    # Обчислення вектора вільних членів B = A * X
    b = np.dot(A, x_exact)

    # Збереження матриці A та вектора B у текстові файли
    np.savetxt(matrix_file, A, fmt="%.16e", delimiter=" ")
    np.savetxt(vector_file, b, fmt="%.16e", delimiter=" ")

    print(f"Матрицю A ({n}x{n}) збережено у '{matrix_file}'.")
    print(f"Вектор B ({n}) збережено у '{vector_file}'.")
    return A, b, x_exact


def read_matrix(filename):
    """
    Зчитує матрицю A з текстового файлу.
    """
    return np.loadtxt(filename, dtype=np.float64)


def read_vector(filename):
    """
    Зчитує вектор B з текстового файлу.
    """
    return np.loadtxt(filename, dtype=np.float64)


def matvec_mul(A, X):
    """
    Обчислює добуток матриці A на вектор X: Y = A * X.
    """
    n = len(X)
    Y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * X[j]
        Y[i] = s
    return Y


def vector_norm(V):
    """
    Обчислює норму Чебишова (максимальну норму) вектора: ||V||_inf = max_i |v_i|.
    """
    max_val = 0.0
    for val in V:
        abs_v = abs(val)
        if abs_v > max_val:
            max_val = abs_v
    return max_val


def matrix_norm(A):
    """
    Обчислює нескінченну норму матриці (максимальну рядкову суму):
    ||A||_inf = max_i sum_j |a_ij|.
    """
    n = A.shape[0]
    max_sum = 0.0
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            row_sum += abs(A[i, j])
        if row_sum > max_sum:
            max_sum = row_sum
    return max_sum


def simple_iteration(A, B, X0, eps=1e-14, max_iter=1000):
    """
    Розв'язок СЛАР методом простої ітерації:
    X^(k+1) = (E - tau * A) * X^(k) + tau * B = X^(k) - tau * (A * X^(k) - B).
    Параметр tau обирається з умови збіжності: 0 < tau < 2 / ||A||.
    """
    n = len(B)
    norm_a = matrix_norm(A)
    # Згідно з методичкою, достатня умова: 0 < tau < 2 / ||A||. Обираємо tau = 1 / ||A||
    tau = 1.0 / norm_a

    X = X0.copy()
    history = []

    for k in range(1, max_iter + 1):
        # Обчислення нового наближення
        AX = matvec_mul(A, X)
        residual = AX - B
        res_norm = vector_norm(residual)

        X_new = X - tau * residual
        diff_norm = vector_norm(X_new - X)

        history.append((k, diff_norm, res_norm))

        if diff_norm < eps:
            return X_new, k, history, tau

        X = X_new

    return X, max_iter, history, tau


def jacobi_method(A, B, X0, eps=1e-14, max_iter=1000):
    """
    Розв'язок СЛАР методом Якобі:
    x_i^(k+1) = (b_i - sum_{j != i} a_ij * x_j^(k)) / a_ii.
    """
    n = len(B)
    X = X0.copy()
    X_new = np.zeros(n, dtype=np.float64)
    history = []

    for k in range(1, max_iter + 1):
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i, j] * X[j]
            X_new[i] = (B[i] - s) / A[i, i]

        diff_norm = vector_norm(X_new - X)
        res_norm = vector_norm(matvec_mul(A, X_new) - B)

        history.append((k, diff_norm, res_norm))

        if diff_norm < eps:
            return X_new.copy(), k, history

        X = X_new.copy()

    return X, max_iter, history


def seidel_method(A, B, X0, eps=1e-14, max_iter=1000):
    """
    Розв'язок СЛАР методом Зейделя:
    x_i^(k+1) = (b_i - sum_{j < i} a_ij * x_j^(k+1) - sum_{j > i} a_ij * x_j^(k)) / a_ii.
    """
    n = len(B)
    X = X0.copy()
    history = []

    for k in range(1, max_iter + 1):
        X_prev = X.copy()
        for i in range(n):
            s1 = 0.0
            for j in range(i):
                s1 += A[i, j] * X[j]
            s2 = 0.0
            for j in range(i + 1, n):
                s2 += A[i, j] * X_prev[j]
            X[i] = (B[i] - s1 - s2) / A[i, i]

        diff_norm = vector_norm(X - X_prev)
        res_norm = vector_norm(matvec_mul(A, X) - B)

        history.append((k, diff_norm, res_norm))

        if diff_norm < eps:
            return X.copy(), k, history

    return X, max_iter, history


def main():
    n = 100
    x_val = 2.5
    matrix_file = "matrix_A.txt"
    vector_file = "vector_B.txt"
    eps = 1e-14

    # 1. Генерація або зчитування СЛАР
    if not (os.path.exists(matrix_file) and os.path.exists(vector_file)):
        A, B, x_exact = generate_system(n=n, x_val=x_val, matrix_file=matrix_file, vector_file=vector_file)
    else:
        A = read_matrix(matrix_file)
        B = read_vector(vector_file)
        x_exact = np.full(n, x_val, dtype=np.float64)

    # 2. Норма матриці та перевірка діагонального переважання
    norm_A = matrix_norm(A)
    min_diag_dominance = min(abs(A[i, i]) - sum(abs(A[i, j]) for j in range(n) if j != i) for i in range(n))
    print(f"Розмірність системи: {n}x{n}")
    print(f"Норма матриці ||A||_inf: {norm_A:.4f}")
    print(f"Мінімальне діагональне переважання min_i (|a_ii| - sum_{{j!=i}} |a_ij|): {min_diag_dominance:.4f}")

    # 3. Початкове наближення (згідно з методичкою: x_i^(0) = 1.0)
    X0 = np.full(n, 1.0, dtype=np.float64)

    # 4. Розв'язання ітераційними методами
    print("\n" + "=" * 70)
    print("РОЗВ'ЯЗАННЯ СЛАР ІТЕРАЦІЙНИМИ МЕТОДАМИ (eps = 1e-14)")
    print("=" * 70)

    # 4.1. Метод простої ітерації
    X_si, it_si, hist_si, tau = simple_iteration(A, B, X0, eps=eps)
    err_si = vector_norm(X_si - x_exact)
    res_si = vector_norm(matvec_mul(A, X_si) - B)
    print(f"\n1. Метод простої ітерації (tau = {tau:.6e}):")
    print(f"   Кількість ітерацій: {it_si}")
    print(f"   Норма приросту кроку ||X^(k) - X^(k-1)||: {hist_si[-1][1]:.2e}")
    print(f"   Норма нев'язки ||A*X - B||:               {res_si:.2e}")
    print(f"   Похибка відносно точного розв'язку:       {err_si:.2e}")

    # 4.2. Метод Якобі
    X_jac, it_jac, hist_jac = jacobi_method(A, B, X0, eps=eps)
    err_jac = vector_norm(X_jac - x_exact)
    res_jac = vector_norm(matvec_mul(A, X_jac) - B)
    print(f"\n2. Метод Якобі:")
    print(f"   Кількість ітерацій: {it_jac}")
    print(f"   Норма приросту кроку ||X^(k) - X^(k-1)||: {hist_jac[-1][1]:.2e}")
    print(f"   Норма нев'язки ||A*X - B||:               {res_jac:.2e}")
    print(f"   Похибка відносно точного розв'язку:       {err_jac:.2e}")

    # 4.3. Метод Зейделя
    X_seid, it_seid, hist_seid = seidel_method(A, B, X0, eps=eps)
    err_seid = vector_norm(X_seid - x_exact)
    res_seid = vector_norm(matvec_mul(A, X_seid) - B)
    print(f"\n3. Метод Зейделя:")
    print(f"   Кількість ітерацій: {it_seid}")
    print(f"   Норма приросту кроку ||X^(k) - X^(k-1)||: {hist_seid[-1][1]:.2e}")
    print(f"   Норма нев'язки ||A*X - B||:               {res_seid:.2e}")
    print(f"   Похибка відносно точного розв'язку:       {err_seid:.2e}")

    print("\n" + "=" * 70)
    print(f"{'Метод':<25} | {'Ітерації':<10} | {'||X^(k)-X^(k-1)||':<18} | {'Похибка':<12}")
    print("-" * 70)
    print(f"{'Проста ітерація':<25} | {it_si:<10} | {hist_si[-1][1]:<18.2e} | {err_si:<12.2e}")
    print(f"{'Якобі':<25} | {it_jac:<10} | {hist_jac[-1][1]:<18.2e} | {err_jac:<12.2e}")
    print(f"{'Зейдель':<25} | {it_seid:<10} | {hist_seid[-1][1]:<18.2e} | {err_seid:<12.2e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
