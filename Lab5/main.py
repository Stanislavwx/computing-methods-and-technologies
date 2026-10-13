import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import special


def f(x):
    """
    Інтенсивність навантаження на сервер у часі x in [0, 24]:
    f(x) = 50 + 20*sin(pi*x/12) + 5*exp(-0.2*(x - 12)^2)
    """
    return 50.0 + 20.0 * np.sin(np.pi * x / 12.0) + 5.0 * np.exp(-0.2 * (x - 12.0)**2)


def exact_integral():
    """
    Точне аналітичне значення інтегралу I0 від f(x) на [0, 24]:
    I0 = int_0^24 50 dx + int_0^24 20*sin(pi*x/12) dx + int_0^24 5*exp(-0.2*(x-12)^2) dx
       = 1200 + 0 + 5*sqrt(5*pi)*erf(12/sqrt(5))
    """
    term1 = 50.0 * 24.0
    term2 = 0.0
    term3 = 5.0 * np.sqrt(5.0 * np.pi) * special.erf(12.0 / np.sqrt(5.0))
    return term1 + term2 + term3


def simpson(f_func, a, b, N):
    """
    Складова квадратурна формула Сімпсона для парного N:
    h = (b - a) / N
    I(N) = (h/3) * [f0 + 4*(f1 + f3 + ...) + 2*(f2 + f4 + ...) + fN]
    """
    if N % 2 != 0:
        raise ValueError("Число розбиттів N має бути парним!")

    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f_func(x)

    # Вагові коефіцієнти: 1 для країв, 4 для непарних індексів, 2 для парних
    s_odd = np.sum(y[1:-1:2])
    s_even = np.sum(y[2:-1:2])

    return (h / 3.0) * (y[0] + 4.0 * s_odd + 2.0 * s_even + y[-1])


def runge_romberg(I_h, I_2h, p=4):
    """
    Уточнення значення інтегралу за схемою Рунге-Ромберга:
    I_R = I(N0) + (I(N0) - I(N0/2)) / (2^p - 1).
    Для методу Сімпсона p = 4, тому дільник дорівнює 15.
    """
    return I_h + (I_h - I_2h) / (2.0**p - 1.0)


def aitken(I_h, I_2h, I_4h):
    """
    Уточнення значення інтегралу за схемою Ейткена та оцінка порядку p:
    I_E = (I_2h^2 - I_h * I_4h) / (2*I_2h - (I_h + I_4h))
    p = (1 / ln(2)) * ln(|(I_4h - I_2h) / (I_2h - I_h)|)
    """
    numerator = I_2h**2 - I_h * I_4h
    denominator = 2.0 * I_2h - (I_h + I_4h)
    I_E = numerator / denominator

    ratio = abs((I_4h - I_2h) / (I_2h - I_h))
    p_order = np.log(ratio) / np.log(2.0)

    return I_E, p_order


def adaptive_simpson(f_func, a, b, delta, max_depth=35):
    """
    Адаптивний алгоритм чисельного інтегрування на основі формули Сімпсона.
    Автоматично дробить відрізок там, де |I1 - I2| > delta_i.
    Повертає: (обчислений інтеграл, число викликів f, розбиття, вузли).
    """
    eval_count = 0
    nodes = []

    def eval_f(x):
        nonlocal eval_count
        eval_count += 1
        nodes.append(x)
        return f_func(x)

    # Початкові 3 точки
    y0 = eval_f(a)
    y1 = eval_f(b)
    x_mid = (a + b) / 2.0
    y_mid = eval_f(x_mid)

    subintervals = []

    def recursive_step(x0, x1, y_start, y_m, y_end, tol, depth):
        h = x1 - x0
        I1 = (h / 6.0) * (y_start + 4.0 * y_m + y_end)

        x_m = (x0 + x1) / 2.0
        x_left = (x0 + x_m) / 2.0
        x_right = (x_m + x1) / 2.0

        y_left = eval_f(x_left)
        y_right = eval_f(x_right)

        I_left = (h / 12.0) * (y_start + 4.0 * y_left + y_m)
        I_right = (h / 12.0) * (y_m + 4.0 * y_right + y_end)
        I2 = I_left + I_right

        # Перевірка умови збіжності
        if abs(I1 - I2) <= tol or depth >= max_depth:
            subintervals.append((x0, x1, h))
            return I2

        # Якщо точність не досягнута, ділимо локальний допуск навпіл
        left_val = recursive_step(x0, x_m, y_start, y_left, y_m, tol / 2.0, depth + 1)
        right_val = recursive_step(x_m, x1, y_m, y_right, y_end, tol / 2.0, depth + 1)
        return left_val + right_val

    I_val = recursive_step(a, b, y0, y_mid, y1, delta, 0)
    return I_val, eval_count, len(subintervals), np.array(sorted(nodes))


def generate_plots(I0, N_opt, eps_opt, N0, eps0, epsR, epsE, p_aitken):
    """Побудова та збереження всіх графічних результатів лабораторної роботи."""
    os.makedirs("plots", exist_ok=True)

    # 1. Графік навантаження на сервер
    x_dense = np.linspace(0, 24, 1000)
    y_dense = f(x_dense)
    y_sine = 50.0 + 20.0 * np.sin(np.pi * x_dense / 12.0)
    y_peak = 5.0 * np.exp(-0.2 * (x_dense - 12.0)**2)

    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(x_dense, y_dense, label=r"$f(x) = 50 + 20\sin(\frac{\pi x}{12}) + 5e^{-0.2(x-12)^2}$",
             color="#1f77b4", linewidth=2.2)
    plt.plot(x_dense, y_sine, "--", label="Базова добова періодика (синусоїда)",
             color="#ff7f0e", linewidth=1.5, alpha=0.8)
    plt.fill_between(x_dense, 50, 50 + y_peak, color="#2ca02c", alpha=0.25,
                     label="Піковий сплеск навантаження (Гаусів пік біля $x = 12$)")
    plt.xlabel("Час, $x$ (години)", fontsize=11)
    plt.ylabel("Інтенсивність навантаження, $f(x)$", fontsize=11)
    plt.title("Добовий профіль навантаження на сервер $f(x)$ на інтервалі [0, 24]", fontsize=12, pad=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(0, 24)
    plt.ylim(25, 75)
    plt.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()
    plt.savefig("plots/server_load_dynamics.png")
    plt.close()

    # 2. Графік збіжності складової формули Сімпсона від N
    Ns = np.arange(10, 1002, 2)
    errs = np.array([abs(simpson(f, 0, 24, n) - I0) for n in Ns])
    # Замінимо нулі на машинний epsilon для log-графіка
    errs_plot = np.where(errs == 0, 1e-15, errs)

    plt.figure(figsize=(9, 5), dpi=300)
    plt.semilogy(Ns, errs_plot, label=r"Похибка $\varepsilon(N) = |I(N) - I_0|$",
                 color="#1f77b4", linewidth=1.8)
    plt.axhline(1e-12, color="#d62728", linestyle="--", linewidth=1.5,
                label=r"Цільовий рівень точності $\varepsilon = 10^{-12}$")
    plt.scatter([N_opt], [eps_opt], color="#d62728", s=85, zorder=5,
                label=f"Оптимальне число розбиттів $N_{{opt}} = {N_opt}$ ($\\sim {eps_opt:.2e}$)")
    plt.scatter([N0], [eps0], color="#9467bd", s=85, zorder=5,
                label=f"Робоча точка $N_0 = {N0}$ ($\\sim {eps0:.2f}$)")
    plt.xlabel("Число розбиттів відрізку, $N$", fontsize=11)
    plt.ylabel("Абсолютна похибка, $\\varepsilon(N)$ (лог. шкала)", fontsize=11)
    plt.title("Залежність похибки складової формули Сімпсона від числа розбиттів $N$", fontsize=12, pad=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.xlim(0, 1000)
    plt.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()
    plt.savefig("plots/simpson_error_convergence.png")
    plt.close()

    # 3. Порівняння методів (Сімпсон, Рунге-Ромберг, Ейткен) для різних N0
    test_N0s = [8, 16, 24, 32]
    e_base_list = []
    e_rr_list = []
    e_aitken_list = []
    p_list = []

    for n in test_N0s:
        i_n = simpson(f, 0, 24, n)
        i_half = simpson(f, 0, 24, n // 2)
        i_qtr = simpson(f, 0, 24, n // 4)
        ir = runge_romberg(i_n, i_half, p=4)
        ie, p_val = aitken(i_n, i_half, i_qtr)

        e_base_list.append(abs(i_n - I0))
        e_rr_list.append(abs(ir - I0))
        e_aitken_list.append(abs(ie - I0))
        p_list.append(p_val)

    fig, ax1 = plt.subplots(figsize=(9, 5.2), dpi=300)
    x_idx = np.arange(len(test_N0s))
    width = 0.26

    r1 = ax1.bar(x_idx - width, e_base_list, width, label="Сімпсон $I(N_0)$", color="#4a7bb0")
    r2 = ax1.bar(x_idx, e_rr_list, width, label="Рунге-Ромберг $I_R$", color="#f28e2b")
    r3 = ax1.bar(x_idx + width, e_aitken_list, width, label="Ейткен $I_E$", color="#59a14f")

    ax1.set_yscale("log")
    ax1.set_xlabel("Робоче число розбиття $N_0$", fontsize=11)
    ax1.set_ylabel("Абсолютна похибка (лог. шкала)", fontsize=11)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([f"$N_0 = {n}$" for n in test_N0s], fontsize=10.5)
    ax1.set_title("Порівняння похибок методів чисельного інтегрування", fontsize=12, pad=12)
    ax1.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/methods_comparison.png")
    plt.close()

    # 4. Дослідження адаптивного алгоритму (похибка та виклики від delta)
    deltas = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
    adapt_errs = []
    adapt_evals = []
    adapt_subs = []

    for d in deltas:
        val, ev, sub, _ = adaptive_simpson(f, 0, 24, d)
        err = abs(val - I0)
        adapt_errs.append(err if err > 0 else 1e-15)
        adapt_evals.append(ev)
        adapt_subs.append(sub)

    fig, (ax_err, ax_ev) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # Лівий графік: похибка від delta
    ax_err.loglog(deltas, adapt_errs, "o-", color="#1f77b4", linewidth=1.8, markersize=6)
    ax_err.loglog(deltas, deltas, "--", color="#d62728", alpha=0.7, label=r"Лінія $\varepsilon = \delta$")
    ax_err.set_xlabel(r"Параметр точності $\delta$", fontsize=11)
    ax_err.set_ylabel(r"Фактична точність $\varepsilon = |I_{adapt} - I_0|$", fontsize=11)
    ax_err.set_title(r"Точність інтегрування від $\delta$", fontsize=11.5)
    ax_err.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_err.legend(fontsize=9.5)

    # Правий графік: обчислення функцій від delta
    ax_ev.loglog(deltas, adapt_evals, "s-", color="#2ca02c", linewidth=1.8, markersize=6)
    ax_ev.set_xlabel(r"Параметр точності $\delta$", fontsize=11)
    ax_ev.set_ylabel("Кількість викликів функції $f(x)$", fontsize=11)
    ax_ev.set_title(r"Обчислювальні витрати від $\delta$", fontsize=11.5)
    ax_ev.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.suptitle("Дослідження характеристик адаптивного алгоритму Сімпсона", fontsize=12.5, y=0.98)
    plt.tight_layout()
    plt.savefig("plots/adaptive_algorithm_analysis.png")
    plt.close()

    # 5. Графік розподілу точок адаптивної сітки (згущення навколо піку)
    _, _, _, sample_nodes = adaptive_simpson(f, 0, 24, 1e-4)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9.5, 5.5), dpi=300, sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1.2]})

    # Верхній графік: функція та положення точок
    ax_top.plot(x_dense, y_dense, color="#1f77b4", linewidth=2, label="Функція $f(x)$")
    ax_top.scatter(sample_nodes, f(sample_nodes), color="#d62728", s=14, alpha=0.6,
                   label=f"Вузли адаптивної сітки ({len(sample_nodes)} точок)")
    ax_top.set_ylabel("Навантаження, $f(x)$", fontsize=10.5)
    ax_top.set_title("Розподіл вузлів адаптивної сітки (згущення в зоні сплеску $x \\approx 12$)",
                     fontsize=11.5, pad=10)
    ax_top.grid(True, linestyle="--", alpha=0.5)
    ax_top.legend(loc="upper right", fontsize=9.5)

    # Нижній графік: гістограма щільності точок
    ax_bot.hist(sample_nodes, bins=24, range=(0, 24), color="#2ca02c", edgecolor="#1b631b", alpha=0.75)
    ax_bot.set_xlabel("Час, $x$ (години)", fontsize=11)
    ax_bot.set_ylabel("Кількість точок", fontsize=10.5)
    ax_bot.grid(True, linestyle="--", alpha=0.5)
    ax_bot.set_xlim(0, 24)

    plt.tight_layout()
    plt.savefig("plots/adaptive_nodes_distribution.png")
    plt.close()


def main():
    print("=" * 80)
    print("ЛАБОРАТОРНА РОБОТА №5: Складова квадратурна формула. Рунге-Ромберг. Ейткен.")
    print("                      Адаптивний алгоритм чисельного інтегрування.")
    print("=" * 80)

    # 1. Постановка: відрізок [0, 24]
    a, b = 0.0, 24.0

    # 2. Точне аналітичне значення інтегралу
    I0 = exact_integral()
    print("\n1-2. Точне значення означеного інтегралу:")
    print(f"     I0 = int_0^24 f(x) dx = 1200 + 5*sqrt(5*pi)*erf(12/sqrt(5))")
    print(f"     I0 = {I0:.15f}")

    # 3. Складова формула Сімпсона: дослідження сітки N in [10, 1000]
    Ns = np.arange(10, 1002, 2)
    errs = np.array([abs(simpson(f, a, b, n) - I0) for n in Ns])

    # 4. Пошук N_opt для заданої точності 1e-12
    target_eps = 1e-12
    idx_opt = np.where(errs <= target_eps)[0][0]
    N_opt = int(Ns[idx_opt])
    eps_opt = float(errs[idx_opt])
    print("\n3-4. Дослідження складової формули Сімпсона (N = 10 ... 1000):")
    print(f"     Цільова точність: eps = {target_eps:.0e}")
    print(f"     Оптимальне число розбиттів N_opt: {N_opt}")
    print(f"     Фактично досягнута точність eps_opt: {eps_opt:.6e}")

    # 5. Обчислення при N0 ≈ N_opt / 10, кратним 8
    # N_opt / 10 = 3.8 -> найближче кратне 8 це N0 = 8
    N0 = 8
    I_N0 = simpson(f, a, b, N0)
    eps0 = abs(I_N0 - I0)
    print(f"\n5. Обчислення базового інтегралу при N0 = {N0} (N0 кратно 8, N0 ≈ N_opt/10):")
    print(f"   I(N0 = {N0}) = {I_N0:.12f}")
    print(f"   Похибка eps0 = |I(N0) - I0| = {eps0:.6e}")

    # 6. Уточнення за методом Рунге-Ромберга (N0 та N0/2)
    I_half = simpson(f, a, b, N0 // 2)
    I_R = runge_romberg(I_N0, I_half, p=4)
    epsR = abs(I_R - I0)
    print("\n6. Уточнення за методом Рунге-Ромберга:")
    print(f"   I(N0/2 = {N0 // 2}) = {I_half:.12f}")
    print(f"   I_R = I(N0) + (I(N0) - I(N0/2)) / 15 = {I_R:.12f}")
    print(f"   Похибка epsR = |I_R - I0| = {epsR:.6e}")

    # 7. Уточнення за методом Ейткена (N0, N0/2, N0/4)
    I_quarter = simpson(f, a, b, N0 // 4)
    I_E, p_est = aitken(I_N0, I_half, I_quarter)
    epsE = abs(I_E - I0)
    print("\n7. Уточнення за методом Ейткена та оцінка порядку точності:")
    print(f"   I(N0/4 = {N0 // 4}) = {I_quarter:.12f}")
    print(f"   I_E = {I_E:.12f}")
    print(f"   Оцінений порядок точності p = {p_est:.6f} (теоретичний p = 4)")
    print(f"   Похибка epsE = |I_E - I0| = {epsE:.6e}")

    # 8. Порівняння результатів для сімейства N0 = [8, 16, 24, 32]
    print("\n8. Порівняльний аналіз методів підвищення точності:")
    print(f"   {'N0':>4} | {'eps0 (Сімпсон)':>16} | {'epsR (Рунге-Ромб.)':>18} | {'epsE (Ейткен)':>16} | {'Порядок p':>10}")
    print("   " + "-" * 72)
    for n in [8, 16, 24, 32]:
        in_val = simpson(f, a, b, n)
        ih_val = simpson(f, a, b, n // 2)
        iq_val = simpson(f, a, b, n // 4)
        ir_val = runge_romberg(in_val, ih_val, p=4)
        ie_val, p_val = aitken(in_val, ih_val, iq_val)
        print(f"   {n:4d} | {abs(in_val - I0):16.6e} | {abs(ir_val - I0):18.6e} | {abs(ie_val - I0):16.6e} | {p_val:10.4f}")

    # 9. Дослідження адаптивного алгоритму
    print("\n9. Дослідження адаптивного алгоритму чисельного інтегрування:")
    print(f"   {'delta':>8} | {'I_adapt':>18} | {'eps_adapt':>12} | {'Виклики f':>10} | {'Підінтервали':>12}")
    print("   " + "-" * 68)
    deltas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    for d in deltas:
        i_ad, ev_c, sub_c, _ = adaptive_simpson(f, a, b, d)
        err_ad = abs(i_ad - I0)
        print(f"   {d:8.0e} | {i_ad:18.12f} | {err_ad:12.4e} | {ev_c:10d} | {sub_c:12d}")

    # 10. Генерація графіків
    generate_plots(I0, N_opt, eps_opt, N0, eps0, epsR, epsE, p_est)
    print("\n[+] Графіки успішно сформовано та збережено у папку plots/:")
    print("    - plots/server_load_dynamics.png")
    print("    - plots/simpson_error_convergence.png")
    print("    - plots/methods_comparison.png")
    print("    - plots/adaptive_algorithm_analysis.png")
    print("    - plots/adaptive_nodes_distribution.png")

    # Збереження результатів у текстовий файл
    with open("results.txt", "w", encoding="utf-8") as rf:
        rf.write("РЕЗУЛЬТАТИ РОЗРАХУНКІВ ЛАБОРАТОРНОЇ РОБОТИ №5\n")
        rf.write("Тема: Складова квадратурна формула. Рунге-Ромберг. Ейткен. Адаптивний алгоритм\n")
        rf.write("=" * 76 + "\n\n")
        rf.write(f"1. Точне значення I0: {I0:.15f}\n\n")
        rf.write(f"2. Дослідження N in [10, 1000]:\n")
        rf.write(f"   - N_opt (для eps <= 1e-12): {N_opt}\n")
        rf.write(f"   - eps_opt: {eps_opt:.6e}\n\n")
        rf.write(f"3. Розрахунки для базового N0 = {N0}:\n")
        rf.write(f"   - I(N0)   = {I_N0:.12f}, eps0 = {eps0:.6e}\n")
        rf.write(f"   - I(N0/2) = {I_half:.12f}\n")
        rf.write(f"   - I(N0/4) = {I_quarter:.12f}\n")
        rf.write(f"   - I_R     = {I_R:.12f}, epsR = {epsR:.6e}\n")
        rf.write(f"   - I_E     = {I_E:.12f}, epsE = {epsE:.6e}, p = {p_est:.6f}\n\n")
        rf.write("4. Зведена таблиця методів при різних N0:\n")
        rf.write(f"{'N0':>4} | {'eps0':>14} | {'epsR':>14} | {'epsE':>14} | {'p':>8}\n")
        rf.write("-" * 62 + "\n")
        for n in [8, 16, 24, 32]:
            in_val = simpson(f, a, b, n)
            ih_val = simpson(f, a, b, n // 2)
            iq_val = simpson(f, a, b, n // 4)
            ir_val = runge_romberg(in_val, ih_val, p=4)
            ie_val, p_val = aitken(in_val, ih_val, iq_val)
            rf.write(f"{n:4d} | {abs(in_val - I0):14.6e} | {abs(ir_val - I0):14.6e} | {abs(ie_val - I0):14.6e} | {p_val:8.4f}\n")
        rf.write("\n5. Адаптивний алгоритм:\n")
        rf.write(f"{'delta':>8} | {'I_adapt':>18} | {'eps':>12} | {'evals':>8} | {'subintervals':>12}\n")
        rf.write("-" * 64 + "\n")
        for d in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
            i_ad, ev_c, sub_c, _ = adaptive_simpson(f, a, b, d)
            rf.write(f"{d:8.0e} | {i_ad:18.12f} | {abs(i_ad - I0):12.4e} | {ev_c:8d} | {sub_c:12d}\n")

    print("\n[+] Результати успішно записано у results.txt")


if __name__ == "__main__":
    main()
