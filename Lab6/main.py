"""
Лабораторна робота №6
Тема: LU-розклад. Ітераційні методи уточнення розв’язку СЛАР.
Дисципліна: Методи та технології обчислень
Виконав: студент 3 курсу групи ФеП-32 Чепара Станіслав
"""

import os
import numpy as np


def generate_system(n=100, x_val=2.5, matrix_file="matrix_A.txt", vector_file="vector_B.txt", seed=26):
    """
    Генерує випадкову матрицю A розмірності n x n та обчислює
    вектор вільних членів B за заданим розв'язком x_i = x_val.
    Результати записуються у текстові файли.
    """
    np.random.seed(seed)
    # Генерація матриці з рівномірним розподілом коефіцієнтів
    A = np.random.uniform(-10.0, 10.0, (n, n))

    # Заданий точний вектор розв'язку
    x_exact = np.full(n, x_val, dtype=np.float64)

    # Обчислення вектора вільних членів B = A * X
    b = np.dot(A, x_exact)

    # Збереження матриці A у текстовий файл
    np.savetxt(matrix_file, A, fmt="%.16e", delimiter=" ")

    # Збереження вектора B у текстовий файл
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


def lu_decomposition(A):
    """
    Обчислює LU-розклад матриці A за схемою Дуліттла:
    A = L * U, де:
      - L: нижня трикутна матриця (l_ij = 0 при j > i);
      - U: верхня трикутна матриця з одиничною головною діагоналлю (u_ii = 1).
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(U, 1.0)

    for k in range(n):
        # Обчислення елементів k-го стовпця матриці L (i >= k)
        for i in range(k, n):
            s = 0.0
            for j in range(k):
                s += L[i, j] * U[j, k]
            L[i, k] = A[i, k] - s

        if abs(L[k, k]) < 1e-15:
            raise ZeroDivisionError(f"Головний елемент L[{k},{k}] близький до нуля.")

        # Обчислення елементів k-го рядка матриці U (i > k)
        for i in range(k + 1, n):
            s = 0.0
            for j in range(k):
                s += L[k, j] * U[j, i]
            U[k, i] = (A[k, i] - s) / L[k, k]

    return L, U


def save_lu_decomposition(L, U, filename="lu_decomposition.txt"):
    """
    Записує матриці L та U у текстовий файл.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Нижня трикутна матриця L:\n")
        np.savetxt(f, L, fmt="%.16e", delimiter=" ")
        f.write("\n# Верхня трикутна матриця U (з одиничною діагоналлю):\n")
        np.savetxt(f, U, fmt="%.16e", delimiter=" ")
    print(f"LU-розклад успішно записано у '{filename}'.")


def solve_lu(L, U, B):
    """
    Розв'язує систему AX = B з використанням LU-розкладу (AX = LUX = B):
    1) Розв'язання LZ = B (прямий хід);
    2) Розв'язання UX = Z (зворотний хід).
    """
    n = len(B)

    # 1. Прямий хід: L * Z = B
    z = np.zeros(n, dtype=np.float64)
    for k in range(n):
        s = 0.0
        for j in range(k):
            s += L[k, j] * z[j]
        z[k] = (B[k] - s) / L[k, k]

    # 2. Зворотний хід: U * X = Z
    x = np.zeros(n, dtype=np.float64)
    for k in range(n - 1, -1, -1):
        s = 0.0
        for j in range(k + 1, n):
            s += U[k, j] * x[j]
        x[k] = z[k] - s

    return x


def matvec_mul(A, X):
    """
    Обчислення добутку матриці A на вектор X: y_i = sum_j a_ij * x_j.
    """
    n = A.shape[0]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * X[j]
        y[i] = s
    return y


def vector_norm(V):
    """
    Обчислення норми Чебишова (максимальної норми) вектора: ||V|| = max_i |v_i|.
    """
    return float(np.max(np.abs(V)))


def iterative_refinement(A, L, U, B, x0, x_exact=None, eps_0=1e-14, max_iter=20):
    """
    Ітераційний метод уточнення розв'язку СЛАР.
    На кожній ітерації:
      1) Обчислюється нев'язка: R = B - A * X^(k)
      2) Розв'язується система для похибки: A * delta_X = R через готовий LU-розклад
      3) Оновлюється наближення: X^(k+1) = X^(k) + delta_X
      4) Перевіряються умови завершення: ||delta_X|| <= eps_0
    """
    x_curr = x0.copy()
    history = []

    # Використовуємо підвищену розрядність для акумуляції нев'язки (метод Вілкінсона),
    # щоб подолати обмеження стандартного заокруглення 64-бітних чисел
    A_ext = A.astype(np.longdouble)
    b_ext = B.astype(np.longdouble)

    for it in range(1, max_iter + 1):
        # 1. Вектор нев'язки R = B - A * X
        r_ext = b_ext - np.dot(A_ext, x_curr.astype(np.longdouble))
        r = r_ext.astype(np.float64)
        res_norm = float(np.max(np.abs(r_ext)))

        # 2. Розв'язання A * delta_X = R через LU-розклад (L*Z = R, U*delta = Z)
        delta_x = solve_lu(L, U, r)
        delta_norm = vector_norm(delta_x)

        # 3. Уточнення розв'язку
        x_curr += delta_x

        # Оцінка похибки відносно точного розв'язку (якщо відомий)
        err_exact = vector_norm(x_curr - x_exact) if x_exact is not None else None

        history.append({
            "iteration": it,
            "delta_norm": delta_norm,
            "residual_norm": res_norm,
            "error_exact": err_exact
        })

        if delta_norm <= eps_0:
            break

    return x_curr, history


def main():
    matrix_file = "matrix_A.txt"
    vector_file = "vector_B.txt"
    lu_file = "lu_decomposition.txt"
    results_file = "results.txt"
    n = 100
    x_target_val = 2.5
    eps_0 = 1e-14

    print("=" * 65)
    print("ЛАБОРАТОРНА РОБОТА №6: LU-РОЗКЛАД ТА ІТЕРАЦІЙНЕ УТОЧНЕННЯ")
    print("=" * 65)

    # 1. Генерація системи або зчитування
    print("\n--- Крок 1. Генерація та збереження СЛАР ---")
    A_gen, b_gen, x_exact = generate_system(
        n=n, x_val=x_target_val,
        matrix_file=matrix_file, vector_file=vector_file,
        seed=26
    )

    # 2. Зчитування матриці та вектора з файлів
    print("\n--- Крок 2. Зчитування матриці A та вектора B з файлів ---")
    A = read_matrix(matrix_file)
    B = read_vector(vector_file)
    print(f"Зчитано матрицю A: розмірність {A.shape}")
    print(f"Зчитано вектор B: розмірність {B.shape}")

    # 3. Знаходження LU-розкладу
    print("\n--- Крок 3. Обчислення LU-розкладу матриці A ---")
    L, U = lu_decomposition(A)
    lu_err = np.max(np.abs(A - np.dot(L, U)))
    print(f"LU-розклад обчислено. Точність розкладу ||A - L*U|| = {lu_err:.3e}")

    # Запис LU-розкладу у файл
    save_lu_decomposition(L, U, lu_file)

    # 4. Розв'язання СЛАР за допомогою LU-розкладу
    print("\n--- Крок 4. Розв'язання системи AX = B за допомогою LU-розкладу ---")
    x0 = solve_lu(L, U, B)

    # Оцінка точності знайденого розв'язку
    Ax0 = matvec_mul(A, x0)
    eps_initial = vector_norm(Ax0 - B)
    err_initial_exact = vector_norm(x0 - x_exact)

    print(f"Точність первинного розв'язку eps = ||A*X0 - B|| = {eps_initial:.6e}")
    print(f"Похибка відносно точного розв'язку ||X0 - X*|| = {err_initial_exact:.6e}")

    # 5. Ітераційне уточнення розв'язку СЛАР
    print(f"\n--- Крок 5. Ітераційне уточнення розв'язку (eps_0 = {eps_0:.0e}) ---")
    x_refined, history = iterative_refinement(
        A, L, U, B, x0, x_exact=x_exact, eps_0=eps_0, max_iter=20
    )

    print(f"{'Ітерація':^10} | {'||ΔX||':^15} | {'||AX - B||':^15} | {'||X - X*||':^15}")
    print("-" * 65)
    for row in history:
        print(f"{row['iteration']:^10} | {row['delta_norm']:^15.6e} | {row['residual_norm']:^15.6e} | {row['error_exact']:^15.6e}")

    final_err_exact = vector_norm(x_refined - x_exact)
    final_res = vector_norm(matvec_mul(A, x_refined) - B)
    total_iters = len(history)

    print("-" * 65)
    print(f"Кількість ітерацій для досягнення точності {eps_0:.0e}: {total_iters}")
    print(f"Фінальна похибка відносно точного розв'язку: {final_err_exact:.6e}")
    print(f"Фінальна нев'язка: {final_res:.6e}")

    # Збереження підсумків у текстовий файл
    with open(results_file, "w", encoding="utf-8") as rf:
        rf.write("ЗВІТ ОБЧИСЛЮВАЛЬНОГО ЕКСПЕРИМЕНТУ (ЛАБОРАТОРНА РОБОТА №6)\n")
        rf.write(f"Розмірність системи: n = {n}\n")
        rf.write(f"Заданий точний розв'язок: x_i = {x_target_val}\n")
        rf.write(f"Точність розкладу ||A - L*U||: {lu_err:.6e}\n\n")
        rf.write("1. Первинний розв'язок (LU-розклад):\n")
        rf.write(f"   - Нев'язка eps = ||A*X0 - B||: {eps_initial:.6e}\n")
        rf.write(f"   - Похибка ||X0 - X*||: {err_initial_exact:.6e}\n\n")
        rf.write(f"2. Ітераційне уточнення розв'язку (цільова точність eps_0 = {eps_0:.0e}):\n")
        rf.write(f"{'Ітерація':<10} {'||ΔX||':<18} {'||AX - B||':<18} {'||X - X*||':<18}\n")
        rf.write("-" * 66 + "\n")
        for row in history:
            rf.write(f"{row['iteration']:<10} {row['delta_norm']:<18.6e} {row['residual_norm']:<18.6e} {row['error_exact']:<18.6e}\n")
        rf.write("-" * 66 + "\n")
        rf.write(f"Необхідна кількість ітерацій: {total_iters}\n")
        rf.write(f"Фінальна похибка розв'язку: {final_err_exact:.6e}\n")

    print(f"\nПідсумкові результати збережено у '{results_file}'.")


if __name__ == "__main__":
    main()
