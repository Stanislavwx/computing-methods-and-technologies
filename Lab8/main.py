import math
import os


# ==============================================================================
# Частина 1. Чисельні методи розв'язування трансцендентних рівнянь
# ==============================================================================

def F(x):
    """
    Трансцендентна функція F(x) = exp(-x) - sin(x).
    """
    return math.exp(-x) - math.sin(x)


def dF(x):
    """
    Перша похідна F'(x) = -exp(-x) - cos(x).
    """
    return -math.exp(-x) - math.cos(x)


def d2F(x):
    """
    Друга похідна F''(x) = exp(-x) + sin(x).
    """
    return math.exp(-x) + math.sin(x)


def tabulate_function(a, b, h, filename="tabulation.txt"):
    """
    Табулює функцію F(x) на проміжку [a, b] з кроком h та записує
    результати у текстовий файл. Повертає список точок перетину з віссю Ox.
    """
    n_steps = int(round((b - a) / h))
    nodes = []
    for i in range(n_steps + 1):
        x = round(a + i * h, 10)
        y = F(x)
        nodes.append((x, y))

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{'x':>10} {'F(x)':>20}\n")
        f.write("-" * 32 + "\n")
        for x, y in nodes:
            f.write(f"{x:10.4f} {y:20.10e}\n")

    # Пошук відрізків зміни знака функції (локалізація коренів)
    roots_info = []
    for i in range(len(nodes) - 1):
        x1, y1 = nodes[i]
        x2, y2 = nodes[i + 1]
        if y1 * y2 <= 0.0:
            x_mid = (x1 + x2) / 2.0
            derivative = dF(x_mid)
            behavior = "зростання" if derivative > 0 else "спадання"
            roots_info.append({
                "interval": (x1, x2),
                "x_approx": x_mid,
                "derivative": derivative,
                "behavior": behavior
            })

    return nodes, roots_info


def simple_iteration(x0, tau, eps=1e-10, max_iter=100):
    """
    Метод простої ітерації (релаксації):
    x_{n+1} = x_n + tau * F(x_n).
    """
    x = x0
    for it in range(1, max_iter + 1):
        fx = F(x)
        x_next = x + tau * fx
        diff = abs(x_next - x)
        res = abs(F(x_next))
        if diff < eps and res < eps:
            return x_next, it, diff, res
        x = x_next
    return x, max_iter, abs(x_next - x), abs(F(x))


def newton_method(x0, eps=1e-10, max_iter=100):
    """
    Метод Ньютона (дотичних):
    x_{n+1} = x_n - F(x_n) / F'(x_n).
    """
    x = x0
    for it in range(1, max_iter + 1):
        fx = F(x)
        dfx = dF(x)
        x_next = x - fx / dfx
        diff = abs(x_next - x)
        res = abs(F(x_next))
        if diff < eps and res < eps:
            return x_next, it, diff, res
        x = x_next
    return x, max_iter, abs(x_next - x), abs(F(x))


def chebyshev_method(x0, eps=1e-10, max_iter=100):
    """
    Метод Чебишева (третього порядку):
    x_{n+1} = x_n - F/F' - 0.5 * (F^2 * F'') / (F'^3).
    """
    x = x0
    for it in range(1, max_iter + 1):
        fx = F(x)
        dfx = dF(x)
        d2fx = d2F(x)
        x_next = x - fx / dfx - 0.5 * (fx**2 * d2fx) / (dfx**3)
        diff = abs(x_next - x)
        res = abs(F(x_next))
        if diff < eps and res < eps:
            return x_next, it, diff, res
        x = x_next
    return x, max_iter, abs(x_next - x), abs(F(x))


def secant_method(x0, x1, eps=1e-10, max_iter=100):
    """
    Метод хорд (січних):
    x_{n+1} = x_n - F(x_n) * (x_n - x_{n-1}) / (F(x_n) - F(x_{n-1})).
    """
    for it in range(1, max_iter + 1):
        f0 = F(x0)
        f1 = F(x1)
        x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
        diff = abs(x_next - x1)
        res = abs(F(x_next))
        if diff < eps and res < eps:
            return x_next, it, diff, res
        x0, x1 = x1, x_next
    return x1, max_iter, abs(x_next - x1), abs(F(x1))


def parabola_method(x0, x1, x2, eps=1e-10, max_iter=100):
    """
    Метод парабол (Мюллера):
    Квадратична інтерполяція за трьома вузлами x_n, x_{n-1}, x_{n-2}.
    """
    for it in range(1, max_iter + 1):
        f0 = F(x0)
        f1 = F(x1)
        f2 = F(x2)

        # Розділені різниці
        f_21 = (f2 - f1) / (x2 - x1)
        f_10 = (f1 - f0) / (x1 - x0)
        f_210 = (f_21 - f_10) / (x2 - x0)

        b = (x2 - x1) * f_210 + f_21
        c = f2
        a = f_210

        discriminant = b**2 - 4.0 * a * c
        if discriminant >= 0:
            d1 = (-b + math.sqrt(discriminant)) / (2.0 * a) if a != 0 else -c / b
            d2 = (-b - math.sqrt(discriminant)) / (2.0 * a) if a != 0 else -c / b
            delta = d1 if abs(d1) < abs(d2) else d2
        else:
            delta = -c / b

        x_next = x2 + delta
        diff = abs(x_next - x2)
        res = abs(F(x_next))
        if diff < eps and res < eps:
            return x_next, it, diff, res
        x0, x1, x2 = x1, x2, x_next
    return x2, max_iter, abs(x_next - x2), abs(F(x2))


def inverse_interpolation_method(x0, x1, x2, eps=1e-10, max_iter=100):
    """
    Метод зворотної інтерполяції за 3 вузлами (поліном Лагранжа x = L(y) при y = 0).
    """
    for it in range(1, max_iter + 1):
        y0 = F(x0)
        y1 = F(x1)
        y2 = F(x2)

        term0 = (y1 * y2) / ((y0 - y1) * (y0 - y2)) * x0
        term1 = (y0 * y2) / ((y1 - y0) * (y1 - y2)) * x1
        term2 = (y0 * y1) / ((y2 - y0) * (y2 - y1)) * x2
        x_next = term0 + term1 + term2

        diff = abs(x_next - x2)
        res = abs(F(x_next))
        if diff < eps and res < eps:
            return x_next, it, diff, res
        x0, x1, x2 = x1, x2, x_next
    return x2, max_iter, abs(x_next - x2), abs(F(x2))


# ==============================================================================
# Частина 2. Чисельні методи розв'язування алгебраїчних рівнянь
# ==============================================================================

def save_polynomial(filename, coeffs):
    """
    Зберігає коефіцієнти многочлена у текстовий файл.
    Коефіцієнти передаються у порядку: [a0, a1, a2, a3].
    """
    with open(filename, "w", encoding="utf-8") as f:
        for c in coeffs:
            f.write(f"{c:.6f}\n")


def read_polynomial(filename):
    """
    Зчитує коефіцієнти многочлена з текстового файлу.
    Повертає список коефіцієнтів [a0, a1, ..., am].
    """
    coeffs = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                coeffs.append(float(line))
    return coeffs


def horner_eval(coeffs, x):
    """
    Обчислює значення многочлена P(x) та його похідної P'(x)
    за схемою Горнера.
    coeffs = [a0, a1, ..., am], степінь m = len(coeffs) - 1.
    """
    m = len(coeffs) - 1
    # 1. Значення многочлена P(x) = b0
    b = [0.0] * (m + 1)
    b[m] = coeffs[m]
    for i in range(m - 1, -1, -1):
        b[i] = coeffs[i] + x * b[i + 1]
    P = b[0]

    # 2. Значення похідної P'(x) = c1
    c = [0.0] * (m + 1)
    c[m] = b[m]
    for i in range(m - 1, 0, -1):
        c[i] = b[i] + x * c[i + 1]
    P_prime = c[1]

    return P, P_prime


def newton_horner(coeffs, x0, eps=1e-10, max_iter=100):
    """
    Знаходження дійсного кореня алгебраїчного рівняння методом Ньютона
    із використанням схеми Горнера.
    """
    x = x0
    for it in range(1, max_iter + 1):
        P, P_prime = horner_eval(coeffs, x)
        x_next = x - P / P_prime
        diff = abs(x_next - x)
        res = abs(P)
        if diff < eps and res < eps:
            return x_next, it, diff, res
        x = x_next
    return x, max_iter, abs(x_next - x), abs(P)


def lin_method(coeffs, alpha0, beta0, eps=1e-10, max_iter=100):
    """
    Знаходження комплексно-спряжених коренів alpha +- i*beta
    алгебраїчного рівняння за методом Ліна.
    Формули строго відповідають методичним вказівкам (ст. 11-12).
    """
    a = coeffs
    for it in range(1, max_iter + 1):
        p0 = -2.0 * alpha0
        q0 = alpha0**2 + beta0**2

        # Ділення многочлена a3*x^3 + a2*x^2 + a1*x + a0 на x^2 + p0*x + q0
        b3 = a[3]
        b2 = a[2] - p0 * b3

        # Формули уточнення коефіцієнтів p1, q1 (ст. 11-12 методички)
        q1 = a[0] / b2
        p1 = (a[1] * b2 - a[0] * b3) / (b2**2)

        alpha1 = -p1 / 2.0
        disc = q1 - alpha1**2
        if disc < 0.0:
            raise ValueError("Дискримінант від'ємний у методі Ліна.")
        beta1 = math.sqrt(disc)

        err = max(abs(alpha1 - alpha0), abs(beta1 - beta0))
        if err < eps:
            return alpha1, beta1, it, err

        alpha0, beta0 = alpha1, beta1

    return alpha0, beta0, max_iter, err


# ==============================================================================
# Головна функція виконання лабораторної роботи
# ==============================================================================

def main():
    print("=" * 80)
    print("ЛАБОРАТОРНА РОБОТА №8: Чисельні методи розв'язування нелінійних рівнянь")
    print("Студент: Чепара Станіслав, група ФеП-32, ЛНУ ім. Івана Франка")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Табуляція трансцендентної функції F(x) = exp(-x) - sin(x) на [0, 4]
    # --------------------------------------------------------------------------
    a, b, h = 0.0, 4.0, 0.1
    eps = 1e-10
    tab_file = "tabulation.txt"
    nodes, roots_info = tabulate_function(a, b, h, filename=tab_file)
    print(f"\n1. Табуляцію функції F(x) = exp(-x) - sin(x) на [{a}, {b}] з кроком {h}")
    print(f"   успішно записано у файл '{tab_file}'.")
    print(f"   Знайдено {len(roots_info)} інтервали зміни знака (ізольованих коренів):")

    for idx, r in enumerate(roots_info, 1):
        x_a, x_b = r["interval"]
        print(f"   Корінь {idx}: відрізок [{x_a:.1f}, {x_b:.1f}], "
              f"x_набл ~ {r['x_approx']:.2f}, F'(x) = {r['derivative']:+.4f} "
              f"(поведінка: {r['behavior']})")

    # --------------------------------------------------------------------------
    # 2-4. Уточнення двох коренів 6 методами до точності eps = 1e-10
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2-4. ДОСЛІДЖЕННЯ ЗБІЖНОСТІ 6 МЕТОДІВ ДЛЯ ТРАНСЦЕНДЕНТНОГО РІВНЯННЯ")
    print("=" * 80)

    # Корінь 1: інтервал [0.5, 0.6], функція спадає (F' < 0)
    # Корінь 2: інтервал [3.0, 3.1], функція зростає (F' > 0)
    test_cases = [
        {
            "name": "Корінь 1 (спадання функції, F' < 0)",
            "approx": 0.55,
            "x0": 0.5,
            "x1": 0.6,
            "x2": 0.55,
            "tau": 0.6  # tau > 0, оскільки F' < 0
        },
        {
            "name": "Корінь 2 (зростання функції, F' > 0)",
            "approx": 3.05,
            "x0": 3.0,
            "x1": 3.1,
            "x2": 3.05,
            "tau": -0.7  # tau < 0, оскільки F' > 0
        }
    ]

    all_results = []
    for tc in test_cases:
        print(f"\n>>> {tc['name']}:")
        x0, x1, x2, tau = tc["x0"], tc["x1"], tc["x2"], tc["tau"]

        # 1. Проста ітерація
        x_si, it_si, diff_si, res_si = simple_iteration(x0, tau, eps=eps)
        # 2. Ньютон
        x_nw, it_nw, diff_nw, res_nw = newton_method(x0, eps=eps)
        # 3. Чебишев
        x_ch, it_ch, diff_ch, res_ch = chebyshev_method(x0, eps=eps)
        # 4. Хорди
        x_sc, it_sc, diff_sc, res_sc = secant_method(x0, x1, eps=eps)
        # 5. Параболи
        x_pb, it_pb, diff_pb, res_pb = parabola_method(x0, x2, x1, eps=eps)
        # 6. Зворотна інтерполяція
        x_ii, it_ii, diff_ii, res_ii = inverse_interpolation_method(x0, x2, x1, eps=eps)

        print(f"{'Метод':<25} | {'Ітерації':<10} | {'Значення кореня':<16} | {'|x_n - x_{n-1}|':<16} | {'|F(x_n)|':<14}")
        print("-" * 87)
        print(f"{'Простої ітерації':<25} | {it_si:<10} | {x_si:<16.10f} | {diff_si:<16.2e} | {res_si:<14.2e}")
        print(f"{'Ньютона':<25} | {it_nw:<10} | {x_nw:<16.10f} | {diff_nw:<16.2e} | {res_nw:<14.2e}")
        print(f"{'Чебишева':<25} | {it_ch:<10} | {x_ch:<16.10f} | {diff_ch:<16.2e} | {res_ch:<14.2e}")
        print(f"{'Хорд':<25} | {it_sc:<10} | {x_sc:<16.10f} | {diff_sc:<16.2e} | {res_sc:<14.2e}")
        print(f"{'Парабол (Мюллера)':<25} | {it_pb:<10} | {x_pb:<16.10f} | {diff_pb:<16.2e} | {res_pb:<14.2e}")
        print(f"{'Зворотної інтерполяції':<25} | {it_ii:<10} | {x_ii:<16.10f} | {diff_ii:<16.2e} | {res_ii:<14.2e}")

        all_results.append({
            "case": tc["name"],
            "root": x_nw,
            "methods": [
                ("Простої ітерації", it_si, diff_si, res_si),
                ("Ньютона", it_nw, diff_nw, res_nw),
                ("Чебишева", it_ch, diff_ch, res_ch),
                ("Хорд", it_sc, diff_sc, res_sc),
                ("Парабол", it_pb, diff_pb, res_pb),
                ("Зворотної інтерполяції", it_ii, diff_ii, res_ii)
            ]
        })

    # --------------------------------------------------------------------------
    # 5-9. Алгебраїчне рівняння 3-го порядку P(x) = x^3 - 4x^2 + 9x - 10 = 0
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5-9. АЛГЕБРАЇЧНЕ РІВНЯННЯ ТРЕТЬОГО ПОРЯДКУ ТА МЕТОДИ ГОРНЕРА І ЛІНА")
    print("=" * 80)
    poly_file = "polynomial.txt"
    # P(x) = (x - 2)(x^2 - 2x + 5) = x^3 - 4x^2 + 9x - 10
    # Коефіцієнти: a0 = -10, a1 = 9, a2 = -4, a3 = 1
    poly_coeffs = [-10.0, 9.0, -4.0, 1.0]
    save_polynomial(poly_file, poly_coeffs)
    read_coeffs = read_polynomial(poly_file)
    print(f"Коефіцієнти многочлена записано у '{poly_file}' та успішно зчитано:")
    print(f"P(x) = {read_coeffs[3]:+.1f}*x^3 {read_coeffs[2]:+.1f}*x^2 {read_coeffs[1]:+.1f}*x {read_coeffs[0]:+.1f} = 0")
    print("Аналітичні корені: x1 = 2.0;  x_{2,3} = 1.0 +- 2.0*i")

    # 8. Дійсний корінь методом Ньютона за схемою Горнера
    x_real, it_horner, diff_h, res_h = newton_horner(read_coeffs, x0=3.0, eps=eps)
    print("\n8. Метод Ньютона за схемою Горнера (дійсний корінь):")
    print(f"   Початкове наближення: x0 = 3.0")
    print(f"   Знайдений корінь:     x  = {x_real:.10f}")
    print(f"   Кількість ітерацій:   k  = {it_horner}")
    print(f"   Крок приросту |dx|:        {diff_h:.2e}")
    print(f"   Нев'язка |P(x)|:           {res_h:.2e}")

    # 9. Комплексні корені за методом Ліна
    alpha0, beta0 = 0.5, 1.5
    alpha_lin, beta_lin, it_lin, err_lin = lin_method(read_coeffs, alpha0, beta0, eps=eps)
    print("\n9. Метод Ліна (комплексно-спряжені корені):")
    print(f"   Початкове наближення: alpha0 = {alpha0}, beta0 = {beta0}")
    print(f"   Знайдені параметри:   alpha = {alpha_lin:.10f}, beta = {beta_lin:.10f}")
    print(f"   Корені рівняння:      x_{{2,3}} = {alpha_lin:.10f} +- {beta_lin:.10f}*i")
    print(f"   Кількість ітерацій:   k = {it_lin}")
    print(f"   Точність (похибка):   {err_lin:.2e}")

    # --------------------------------------------------------------------------
    # Збереження підсумкових результатів у results.txt
    # --------------------------------------------------------------------------
    results_file = "results.txt"
    with open(results_file, "w", encoding="utf-8") as rf:
        rf.write("ПІДСУМКОВІ РЕЗУЛЬТАТИ ЛАБОРАТОРНОЇ РОБОТИ №8\n")
        rf.write("=" * 60 + "\n\n")
        rf.write("1. Трансцендентне рівняння: F(x) = exp(-x) - sin(x) = 0\n")
        rf.write(f"Точність зупинки: eps = {eps}\n\n")
        for res in all_results:
            rf.write(f"{res['case']} (x* = {res['root']:.10f}):\n")
            rf.write(f"{'Метод':<25} | {'Ітерації':<8} | {'|x_n - x_{n-1}|':<15} | {'|F(x_n)|':<12}\n")
            rf.write("-" * 65 + "\n")
            for m_name, it_cnt, d_val, r_val in res["methods"]:
                rf.write(f"{m_name:<25} | {it_cnt:<8} | {d_val:<15.2e} | {r_val:<12.2e}\n")
            rf.write("\n")

        rf.write("2. Алгебраїчне рівняння: P(x) = x^3 - 4x^2 + 9x - 10 = 0\n")
        rf.write(f"Дійсний корінь (Ньютон-Горнер): x = {x_real:.10f}, ітерацій: {it_horner}, |P| = {res_h:.2e}\n")
        rf.write(f"Комплексні корені (Лін): x = {alpha_lin:.10f} +- {beta_lin:.10f}*i, ітерацій: {it_lin}, похибка: {err_lin:.2e}\n")

    print(f"\nПідсумкові результати також збережено у '{results_file}'.")
    print("=" * 80)


if __name__ == "__main__":
    main()
