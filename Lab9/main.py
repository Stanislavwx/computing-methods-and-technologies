"""
Лабораторна робота №9
Тема: Метод Хука-Дживса багатовимірної оптимізації для розв’язку системи нелінійних рівнянь
Дисципліна: Методи та технології обчислень
Виконав: студент 3 курсу групи ФеП-32 Чепара Станіслав
Викладач: асистент Каськун О. Д.
Львів — 2026
"""

import math
import os


# ==============================================================================
# 1. Цільові функції та система нелінійних рівнянь
# ==============================================================================

def rosenbrock(x):
    """
    Класична тестова яружна функція Розенброка:
    f(x1, x2) = 100 * (x1^2 - x2)^2 + (x1 - 1)^2.
    Мінімум: f(1.0, 1.0) = 0.
    """
    return 100.0 * (x[0]**2 - x[1])**2 + (x[0] - 1.0)**2


def sys_f1(x1, x2):
    """
    Перше нелінійне рівняння системи (коло радіуса R = 2):
    f1(x1, x2) = x1^2 + x2^2 - 4 = 0.
    """
    return x1**2 + x2**2 - 4.0


def sys_f2(x1, x2):
    """
    Друге нелінійне рівняння системи (синусоїда, зміщена вгору на 1):
    f2(x1, x2) = x2 - sin(x1) - 1 = 0.
    """
    return x2 - math.sin(x1) - 1.0


def system_phi(x):
    """
    Допоміжна цільова функція для розв'язання системи нелінійних рівнянь (m = 2):
    Phi(x1, x2) = f1(x1, x2)^2 + f2(x1, x2)^2.
    У точці кореня системи Phi(x*) = 0.
    """
    f1 = sys_f1(x[0], x[1])
    f2 = sys_f2(x[0], x[1])
    return f1**2 + f2**2


# ==============================================================================
# 2. Реалізація методу Хука-Дживса (Hooke-Jeeves)
# ==============================================================================

def hooke_jeeves(func, x0, dx0, q=2.0, p=1.0, use_pattern=True,
                 eps1=1e-6, eps2=1e-8, max_iter=1000):
    """
    Чисельний метод нульового порядку Хука-Дживса для багатовимірної оптимізації.

    Параметри:
      func        - цільова функція func(x), x - список координат
      x0          - початкове наближення (базисна точка)
      dx0         - початковий вектор кроків по кожній координаті
      q           - коефіцієнт дроблення (зменшення) кроку (q > 1)
      p           - множник екстраполяції для пошуку по зразку
      use_pattern - прапорець використання етапу пошуку по зразку
      eps1        - критерій закінчення за величиною кроку/приросту координат
      eps2        - критерій закінчення за зміною цільової функції
      max_iter    - гранична кількість ітерацій (захист від зациклення)

    Повертає:
      x_best      - знайдена точка мінімуму
      f_best      - значення цільової функції у точці мінімуму
      step_count  - кількість кроків оптимізації
      func_evals  - загальна кількість обчислень цільової функції
      trajectory  - список записів траєкторії спуску
      status_msg  - текстовий опис причини зупинки
    """
    n = len(x0)
    x_base = [float(v) for v in x0]
    dx = [float(d) for d in dx0]

    trajectory = []
    func_evals = 1
    f_curr = func(x_base)

    trajectory.append({
        'step': 0,
        'stage': 'Початкова точка',
        'x': list(x_base),
        'dx': list(dx),
        'f': f_curr
    })

    step_count = 0

    while step_count < max_iter:
        step_count += 1
        x_curr = list(x_base)
        f_base = f_curr

        # --- Етап 1: Досліджуючий пошук навколо базисної точки ---
        for i in range(n):
            improved = False

            # Спроба кроку в додатному напрямку +dx[i]
            x_try = list(x_curr)
            x_try[i] = x_base[i] + dx[i]
            f_try = func(x_try)
            func_evals += 1
            if f_try < f_curr:
                x_curr = x_try
                f_curr = f_try
                improved = True
            else:
                # Спроба кроку у від'ємному напрямку -dx[i]
                x_try = list(x_curr)
                x_try[i] = x_base[i] - dx[i]
                f_try = func(x_try)
                func_evals += 1
                if f_try < f_curr:
                    x_curr = x_try
                    f_curr = f_try
                    improved = True

            # Якщо успіху немає — дробимо крок по координаті i до досягнення eps1
            if not improved:
                cur_d = dx[i]
                while cur_d >= eps1:
                    cur_d /= q
                    # Проба +
                    x_try = list(x_curr)
                    x_try[i] = x_base[i] + cur_d
                    f_try = func(x_try)
                    func_evals += 1
                    if f_try < f_curr:
                        x_curr = x_try
                        f_curr = f_try
                        dx[i] = cur_d
                        improved = True
                        break

                    # Проба -
                    x_try = list(x_curr)
                    x_try[i] = x_base[i] - cur_d
                    f_try = func(x_try)
                    func_evals += 1
                    if f_try < f_curr:
                        x_curr = x_try
                        f_curr = f_try
                        dx[i] = cur_d
                        improved = True
                        break

                if not improved:
                    x_curr[i] = x_base[i]

        trajectory.append({
            'step': step_count,
            'stage': 'Досліджуючий пошук',
            'x': list(x_curr),
            'dx': list(dx),
            'f': f_curr
        })

        norm_dx = math.sqrt(sum(d**2 for d in dx))
        diff_f = abs(f_curr - f_base)

        # Перевірка умов завершення
        if x_curr == x_base:
            return x_base, f_base, step_count, func_evals, trajectory, 'Локальний мінімум (крок вичерпано)'

        if norm_dx < eps1 and diff_f < eps2:
            return x_curr, f_curr, step_count, func_evals, trajectory, 'Досягнуто задану точність (||dx|| < eps1, |df| < eps2)'

        if not use_pattern:
            x_base = x_curr
            continue

        # --- Етап 2: Пошук по зразку (Pattern Search) ---
        while step_count < max_iter:
            # Точка екстраполяції за напрямком зразка
            x_p = [x_curr[j] + p * (x_curr[j] - x_base[j]) for j in range(n)]
            f_p = func(x_p)
            func_evals += 1

            # Досліджуючий пошук з точки x_p без зменшення величин кроку
            x_p_expl = list(x_p)
            f_p_expl = f_p
            for i in range(n):
                x_try = list(x_p_expl)
                x_try[i] += dx[i]
                f_try = func(x_try)
                func_evals += 1
                if f_try < f_p_expl:
                    x_p_expl = x_try
                    f_p_expl = f_try
                else:
                    x_try = list(x_p_expl)
                    x_try[i] -= dx[i]
                    f_try = func(x_try)
                    func_evals += 1
                    if f_try < f_p_expl:
                        x_p_expl = x_try
                        f_p_expl = f_try

            # Перевірка: чи зменшилась цільова функція порівняно з поточною базовою
            if f_p_expl < f_curr:
                step_count += 1
                x_base = x_curr
                x_curr = x_p_expl
                f_curr = f_p_expl
                trajectory.append({
                    'step': step_count,
                    'stage': 'Пошук по зразку',
                    'x': list(x_curr),
                    'dx': list(dx),
                    'f': f_curr
                })

                if math.sqrt(sum(d**2 for d in dx)) < eps1 and abs(f_curr - func(x_base)) < eps2:
                    return x_curr, f_curr, step_count, func_evals, trajectory, 'Досягнуто точність під час руху по зразку'
            else:
                # Рух по зразку не приніс поліпшення — повернення до досліджуючого пошуку
                x_base = x_curr
                break

    return x_curr, f_curr, step_count, func_evals, trajectory, 'Перевищено ліміт ітерацій'


# ==============================================================================
# 3. Допоміжні функції для збереження результатів у файли
# ==============================================================================

def save_trajectory_to_file(filename, header_title, trajectory):
    """
    Записує покрокову траєкторію спуску у текстовий файл.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {header_title}\n")
        f.write(f"# {'Крок':>5} | {'Етап пошуку':<22} | {'x1':>14} | {'x2':>14} | {'dx1':>12} | {'dx2':>12} | {'Значення f(X)':>16}\n")
        f.write("# " + "-" * 105 + "\n")
        for entry in trajectory:
            st = entry['step']
            stage = entry['stage']
            x = entry['x']
            dx = entry['dx']
            f_val = entry['f']
            f.write(f"{st:7d} | {stage:<22} | {x[0]:14.8f} | {x[1]:14.8f} | {dx[0]:12.4e} | {dx[1]:12.4e} | {f_val:16.8e}\n")


# ==============================================================================
# 4. Основна обчислювальна процедура
# ==============================================================================

def main():
    print("=" * 80)
    print("ЛАБОРАТОРНА РОБОТА №9: МЕТОД ХУКА-ДЖИВСА БАГАТОВИМІРНОЇ ОПТИМІЗАЦІЇ")
    print("Дисципліна: Методи та технології обчислень")
    print("Студент: Чепара Станіслав (група ФеП-32)")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Частина 1. Тестування методу на функції Розенброка
    # --------------------------------------------------------------------------
    print("\n--- ЧАСТИНА 1. Тестування алгоритму на функції Розенброка ---")
    x0_rosen = [-1.2, 0.0]
    dx0_rosen = [0.5, 0.5]
    q_val = 2.0
    p_val = 1.0
    eps1_val = 1e-5
    eps2_val = 1e-7

    xr, fr, sr, fer, traj_r, msg_r = hooke_jeeves(
        rosenbrock, x0_rosen, dx0_rosen,
        q=q_val, p=p_val, eps1=eps1_val, eps2=eps2_val
    )

    print(f"Початкова точка:        X(0) = {x0_rosen}")
    print(f"Початковий крок:        dX   = {dx0_rosen}")
    print(f"Коефіцієнти:            q = {q_val}, p = {p_val}")
    print(f"Точності зупинки:       eps1 = {eps1_val}, eps2 = {eps2_val}")
    print(f"Знайдений мінімум:      X* = ({xr[0]:.8f}, {xr[1]:.8f})")
    print(f"Значення функції:       f(X*) = {fr:.8e}")
    print(f"Кількість кроків:       {sr}")
    print(f"Кількість викликів f:   {fer}")
    print(f"Статус зупинки:         {msg_r}")

    save_trajectory_to_file("trajectory_test.txt",
                            "Траєкторія оптимізації функції Розенброка", traj_r)
    print("-> Траєкторію спуску збережено у файл 'trajectory_test.txt'")

    # --------------------------------------------------------------------------
    # Частина 2. Розв'язок системи нелінійних рівнянь (m = 2)
    # --------------------------------------------------------------------------
    print("\n--- ЧАСТИНА 2. Розв'язання системи нелінійних рівнянь (m = 2) ---")
    print("Система рівнянь:")
    print("  f1(x1, x2) = x1^2 + x2^2 - 4 = 0     (коло R = 2)")
    print("  f2(x1, x2) = x2 - sin(x1) - 1 = 0    (синусоїда, зміщена на 1)")
    print("Цільова функція: Phi(X) = f1^2 + f2^2")

    # Корінь 1 (перший квадрант)
    x0_sys1 = [1.0, 1.5]
    dx0_sys = [0.2, 0.2]
    eps1_sys = 1e-7
    eps2_sys = 1e-10

    x1_opt, phi1_opt, s1, fe1, traj_s1, msg_s1 = hooke_jeeves(
        system_phi, x0_sys1, dx0_sys,
        q=2.0, p=1.0, eps1=eps1_sys, eps2=eps2_sys
    )
    res_f1_root1 = sys_f1(x1_opt[0], x1_opt[1])
    res_f2_root1 = sys_f2(x1_opt[0], x1_opt[1])

    print(f"\n[Корінь 1]")
    print(f"  Початкове наближення: X(0) = {x0_sys1}")
    print(f"  Знайдений розв'язок:  X1* = ({x1_opt[0]:.8f}, {x1_opt[1]:.8f})")
    print(f"  Значення Phi(X*):     {phi1_opt:.8e}")
    print(f"  Нев'язка f1:          {res_f1_root1:.8e}")
    print(f"  Нев'язка f2:          {res_f2_root1:.8e}")
    print(f"  Кількість кроків:     {s1} (обчислень Phi: {fe1})")
    print(f"  Статус:               {msg_s1}")

    # Корінь 2 (другий квадрант)
    x0_sys2 = [-1.8, 0.2]
    x2_opt, phi2_opt, s2, fe2, traj_s2, msg_s2 = hooke_jeeves(
        system_phi, x0_sys2, dx0_sys,
        q=2.0, p=1.0, eps1=eps1_sys, eps2=eps2_sys
    )
    res_f1_root2 = sys_f1(x2_opt[0], x2_opt[1])
    res_f2_root2 = sys_f2(x2_opt[0], x2_opt[1])

    print(f"\n[Корінь 2]")
    print(f"  Початкове наближення: X(0) = {x0_sys2}")
    print(f"  Знайдений розв'язок:  X2* = ({x2_opt[0]:.8f}, {x2_opt[1]:.8f})")
    print(f"  Значення Phi(X*):     {phi2_opt:.8e}")
    print(f"  Нев'язка f1:          {res_f1_root2:.8e}")
    print(f"  Нев'язка f2:          {res_f2_root2:.8e}")
    print(f"  Кількість кроків:     {s2} (обчислень Phi: {fe2})")
    print(f"  Статус:               {msg_s2}")

    # Запис траєкторій системи у спільний файл
    with open("trajectory_system.txt", "w", encoding="utf-8") as f:
        f.write("# ТРАЄКТОРІЇ СПУСКУ ДЛЯ РОЗВ'ЯЗАННЯ СИСТЕМИ НЕЛІНІЙНИХ РІВНЯНЬ\n\n")
        f.write("# === КОРІНЬ 1 ===\n")
        f.write(f"# {'Крок':>5} | {'Етап пошуку':<22} | {'x1':>14} | {'x2':>14} | {'dx1':>12} | {'dx2':>12} | {'Значення Phi(X)':>16}\n")
        f.write("# " + "-" * 105 + "\n")
        for entry in traj_s1:
            st = entry['step']
            stage = entry['stage']
            x = entry['x']
            dx = entry['dx']
            f_val = entry['f']
            f.write(f"{st:7d} | {stage:<22} | {x[0]:14.8f} | {x[1]:14.8f} | {dx[0]:12.4e} | {dx[1]:12.4e} | {f_val:16.8e}\n")

        f.write("\n\n# === КОРІНЬ 2 ===\n")
        f.write(f"# {'Крок':>5} | {'Етап пошуку':<22} | {'x1':>14} | {'x2':>14} | {'dx1':>12} | {'dx2':>12} | {'Значення Phi(X)':>16}\n")
        f.write("# " + "-" * 105 + "\n")
        for entry in traj_s2:
            st = entry['step']
            stage = entry['stage']
            x = entry['x']
            dx = entry['dx']
            f_val = entry['f']
            f.write(f"{st:7d} | {stage:<22} | {x[0]:14.8f} | {x[1]:14.8f} | {dx[0]:12.4e} | {dx[1]:12.4e} | {f_val:16.8e}\n")

    print("-> Траєкторії спуску для системи збережено у файл 'trajectory_system.txt'")

    # --------------------------------------------------------------------------
    # Частина 3. Дослідження впливу параметрів алгоритму (результати)
    # --------------------------------------------------------------------------
    print("\n--- ЧАСТИНА 3. Параметричні дослідження ефективності ---")
    
    # 3.1. Порівняння ролі пошуку по зразку
    _, f_p, s_p, fe_p, _, _ = hooke_jeeves(rosenbrock, x0_rosen, dx0_rosen, q=2.0, p=1.0, use_pattern=True)
    _, f_np, s_np, fe_np, _, _ = hooke_jeeves(rosenbrock, x0_rosen, dx0_rosen, q=2.0, p=1.0, use_pattern=False)
    print(f"Функція Розенброка з пошуком по зразку:    кроків={s_p}, викликів={fe_p}, f={f_p:.2e}")
    print(f"Функція Розенброка без пошуку по зразку:  кроків={s_np}, викликів={fe_np}, f={f_np:.2e}")

    # 3.2. Дослідження коефіцієнта дроблення кроку q
    q_experiments = []
    for q_test in [1.5, 2.0, 3.0, 4.0]:
        x_res, phi_res, s_res, fe_res, _, _ = hooke_jeeves(system_phi, x0_sys1, [0.2, 0.2], q=q_test, p=1.0)
        q_experiments.append((q_test, s_res, fe_res, phi_res))
        print(f"  q = {q_test:<3}: кроків = {s_res:3d}, викликів = {fe_res:4d}, Phi = {phi_res:.2e}")

    # 3.3. Дослідження початкової величини кроку dx
    dx_experiments = []
    for dx_test in [0.05, 0.1, 0.2, 0.5]:
        x_res, phi_res, s_res, fe_res, _, _ = hooke_jeeves(system_phi, x0_sys1, [dx_test, dx_test], q=2.0, p=1.0)
        dx_experiments.append((dx_test, s_res, fe_res, phi_res))
        print(f"  dx = {dx_test:<4}: кроків = {s_res:3d}, викликів = {fe_res:4d}, Phi = {phi_res:.2e}")

    # Запис зведених результатів у файл results.txt
    with open("results.txt", "w", encoding="utf-8") as rf:
        rf.write("================================================================================\n")
        rf.write("ПІДСУМКОВІ РЕЗУЛЬТАТИ ОБЧИСЛЮВАЛЬНИХ ЕКСПЕРИМЕНТІВ (ЛАБОРАТОРНА РОБОТА №9)\n")
        rf.write("Метод Хука-Дживса багатовимірної оптимізації\n")
        rf.write("Виконав: студент групи ФеП-32 Чепара Станіслав\n")
        rf.write("================================================================================\n\n")

        rf.write("1. ТЕСТОВА ФУНКЦІЯ РОЗЕНБРОКА:\n")
        rf.write(f"   Початкова точка:     X(0) = {x0_rosen}\n")
        rf.write(f"   Початковий крок:     dX   = {dx0_rosen}\n")
        rf.write(f"   Параметри пошуку:    q = {q_val}, p = {p_val}, eps1 = {eps1_val}, eps2 = {eps2_val}\n")
        rf.write(f"   Знайдений мінімум:   X*   = ({xr[0]:.10f}, {xr[1]:.10f})\n")
        rf.write(f"   Значення функції:    f(X*) = {fr:.10e}\n")
        rf.write(f"   Число кроків:        {sr}\n")
        rf.write(f"   Число викликів f:    {fer}\n")
        rf.write(f"   Ефект прискорення:   зразковий крок скоротив число ітерацій з {s_np} до {sr}\n\n")

        rf.write("2. СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ (m = 2):\n")
        rf.write("   Рівняння 1: f1(x1, x2) = x1^2 + x2^2 - 4 = 0\n")
        rf.write("   Рівняння 2: f2(x1, x2) = x2 - sin(x1) - 1 = 0\n")
        rf.write("   Цільова функція: Phi(X) = f1^2 + f2^2\n\n")

        rf.write("   [Корінь 1 (I квадрант)]:\n")
        rf.write(f"     Початкове наближення: X(0) = {x0_sys1}\n")
        rf.write(f"     Знайдений розв'язок:  X1*  = ({x1_opt[0]:.10f}, {x1_opt[1]:.10f})\n")
        rf.write(f"     Значення Phi(X*):     {phi1_opt:.10e}\n")
        rf.write(f"     Нев'язка f1:          {res_f1_root1:.10e}\n")
        rf.write(f"     Нев'язка f2:          {res_f2_root1:.10e}\n")
        rf.write(f"     Кількість кроків:     {s1} (обчислень Phi: {fe1})\n\n")

        rf.write("   [Корінь 2 (II квадрант)]:\n")
        rf.write(f"     Початкове наближення: X(0) = {x0_sys2}\n")
        rf.write(f"     Знайдений розв'язок:  X2*  = ({x2_opt[0]:.10f}, {x2_opt[1]:.10f})\n")
        rf.write(f"     Значення Phi(X*):     {phi2_opt:.10e}\n")
        rf.write(f"     Нев'язка f1:          {res_f1_root2:.10e}\n")
        rf.write(f"     Нев'язка f2:          {res_f2_root2:.10e}\n")
        rf.write(f"     Кількість кроків:     {s2} (обчислень Phi: {fe2})\n\n")

        rf.write("3. ПАРАМЕТРИЧНИЙ АНАЛІЗ ЗБІЖНОСТІ (для Кореня 1):\n")
        rf.write("   Вплив коефіцієнта дроблення q (при dX = 0.2):\n")
        for q_t, s_t, fe_t, phi_t in q_experiments:
            rf.write(f"     q = {q_t:<3} -> кроків: {s_t:3d}, обчислень: {fe_t:4d}, Phi = {phi_t:.2e}\n")
        rf.write("\n   Вплив початкового кроку dX (при q = 2.0):\n")
        for dx_t, s_t, fe_t, phi_t in dx_experiments:
            rf.write(f"     dX = {dx_t:<4} -> кроків: {s_t:3d}, обчислень: {fe_t:4d}, Phi = {phi_t:.2e}\n")

    print("-> Підсумкові результати збережено у файл 'results.txt'")
    print("\nОбчислення успішно завершено.")


if __name__ == "__main__":
    main()
