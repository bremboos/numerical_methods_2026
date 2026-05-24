import numpy as np
import matplotlib.pyplot as plt


# ТРАНСЦЕНДЕНТНІ РІВНЯННЯ

def f_transcendent(x):
    return np.sin(x) - 0.5


def df_transcendent(x):
    return np.cos(x)


def d2f_transcendent(x):
    return -np.sin(x)



def is_converged(x_new, x_old, eps=1e-10):
    return abs(f_transcendent(x_new)) < eps and abs(x_new - x_old) < eps


# --- Реалізація 6 методів уточнення коренів із критерієм зупинки ---

def simple_iteration(x0, max_iter=12):
    tau = 0.8 if df_transcendent(x0) > 0 else -0.8
    history = [abs(f_transcendent(x0))]
    x = x0
    for _ in range(max_iter):
        x_next = x + tau * f_transcendent(x)
        history.append(abs(f_transcendent(x_next)))

        if is_converged(x_next, x):
            x = x_next
            break

        x = x_next
    return x, history


def newton_method(x0, max_iter=4):
    history = [abs(f_transcendent(x0))]
    x = x0
    for _ in range(max_iter):
        x_next = x - f_transcendent(x) / df_transcendent(x)
        history.append(abs(f_transcendent(x_next)))


        if is_converged(x_next, x):
            x = x_next
            break

        x = x_next
    return x, history


def chebyshev_method(x0, max_iter=4):
    history = [abs(f_transcendent(x0))]
    x = x0
    for _ in range(max_iter):
        f_val = f_transcendent(x)
        df_val = df_transcendent(x)
        d2f_val = d2f_transcendent(x)
        x_next = x - f_val / df_val - 0.5 * (f_val ** 2 * d2f_val) / (df_val ** 3)
        history.append(abs(f_transcendent(x_next)))

        if is_converged(x_next, x):
            x = x_next
            break

        x = x_next
    return x, history


def secant_method(x0, x1, max_iter=6):
    history = [abs(f_transcendent(x0)), abs(f_transcendent(x1))]
    x_prev, x_curr = x0, x1
    for _ in range(max_iter - 1):
        f1, f0 = f_transcendent(x_curr), f_transcendent(x_prev)
        if f1 - f0 == 0:
            break
        x_next = x_curr - f1 * (x_curr - x_prev) / (f1 - f0)
        history.append(abs(f_transcendent(x_next)))

        if is_converged(x_next, x_curr):
            x_curr = x_next
            break

        x_prev, x_curr = x_curr, x_next
    return x_curr, history


def muller_method(x0, x1, x2, max_iter=6):
    history = [abs(f_transcendent(x0)), abs(f_transcendent(x1)), abs(f_transcendent(x2))]
    xa, xb, xc = x0, x1, x2
    for _ in range(max_iter - 2):
        f0, f1, f2 = f_transcendent(xa), f_transcendent(xb), f_transcendent(xc)
        f_10 = (f1 - f0) / (xb - xa)
        f_21 = (f2 - f1) / (xc - xb)
        f_210 = (f_21 - f_10) / (xc - xa)
        w = f_21 + (xc - xb) * f_210
        det = np.sqrt(max(0.0, w ** 2 - 4 * f2 * f_210))
        den = w + det if abs(w + det) > abs(w - det) else w - det
        if den == 0:
            break
        x_next = xc - 2 * f2 / den
        history.append(abs(f_transcendent(x_next)))

        if is_converged(x_next, xc):
            xc = x_next
            break

        xa, xb, xc = xb, xc, x_next
    return xc, history


def inverse_interpolation(x0, x1, x2, max_iter=6):
    history = [abs(f_transcendent(x0)), abs(f_transcendent(x1)), abs(f_transcendent(x2))]
    xa, xb, xc = x0, x1, x2
    for _ in range(max_iter - 2):
        y0, y1, y2 = f_transcendent(xa), f_transcendent(xb), f_transcendent(xc)
        if (y0 - y1) == 0 or (y0 - y2) == 0 or (y1 - y2) == 0:
            break
        x_next = (
                (y1 * y2) / ((y0 - y1) * (y0 - y2)) * xa +
                (y0 * y2) / ((y1 - y0) * (y1 - y2)) * xb +
                (y0 * y1) / ((y2 - y0) * (y2 - y1)) * xc
        )
        history.append(abs(f_transcendent(x_next)))

        if is_converged(x_next, xc):
            xc = x_next
            break

        xa, xb, xc = xb, xc, x_next
    return xc, history


# Функція побудови графіків збіжності
def plot_convergence(title, init_points):
    plt.figure(figsize=(10, 6))

    _, h_si = simple_iteration(init_points[2], max_iter=9)
    _, h_nr = newton_method(init_points[2], max_iter=2)
    _, h_ch = chebyshev_method(init_points[2], max_iter=2)
    _, h_sec = secant_method(init_points[1], init_points[2], max_iter=5)
    _, h_mul = muller_method(init_points[0], init_points[1], init_points[2], max_iter=5)
    _, h_inv = inverse_interpolation(init_points[0], init_points[1], init_points[2], max_iter=5)

    plt.plot(h_si, 'o-', color='#2ca02c', linewidth=1.8, label="Метод простої ітерації")
    plt.plot(h_nr, 's-', color='#ff7f0e', linewidth=1.8, label="Метод Ньютона")
    plt.plot(h_ch, '^-', color='#d62728', linewidth=1.8, label="Метод Чебишева")
    plt.plot(h_sec, 'd-', color='#9467bd', linewidth=1.8, label="Метод хорд")
    plt.plot(h_mul, 'v-', color='#17becf', linewidth=1.8, label="Метод парабол")
    plt.plot(h_inv, 'x-', color='#8c564b', linewidth=1.8, label="Метод зворотної інтеполяції")

    plt.yscale('log')
    plt.title(title, fontsize=12, pad=12)
    plt.xlabel("Номер наближення", fontsize=10)
    plt.ylabel("|F(x_n)|", fontsize=10)
    plt.grid(True, which="both", ls="--", color='lightgray', alpha=0.7)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.show()


#АЛГЕБРАЇЧНІ РІВНЯННЯ

def p_algebraic(coefficients_list, x):
    return (coefficients_list[0] * x ** 3 + coefficients_list[1] * x ** 2 +
            coefficients_list[2] * x + coefficients_list[3])


# Обчислення P(x) та P'(x) за схемою Горнера
def horner_newton_step(coefficients_list, xn):
    b3 = coefficients_list[0]
    b2 = coefficients_list[1] + xn * b3
    b1 = coefficients_list[2] + xn * b2
    b0 = coefficients_list[3] + xn * b1

    c3 = b3
    c2 = b2 + xn * c3
    c1 = b1 + xn * c2

    return b0, c1


# Метод Ньютона для дійсного кореня
def solve_real_horner(coefficients_list, x0, eps=1e-10):
    x_val = x0
    for i in range(1, 100):
        p_val, dp_val = horner_newton_step(coefficients_list, x_val)
        if dp_val == 0:
            break
        x_next = x_val - p_val / dp_val

        if abs(p_val) < eps and abs(x_next - x_val) < eps:
            return x_next, i

        x_val = x_next
    return x_val, 100


# Метод Ліна для комплексних коренів
def solve_complex_lin(coefficients_list, alpha0, beta0, eps=1e-10):
    a3, a2, a1, a0 = coefficients_list
    alpha_val, beta_val = alpha0, beta0
    for i in range(1, 100):
        p_param = -2 * alpha_val
        q_param = alpha_val ** 2 + beta_val ** 2

        b3 = a3
        b2 = a2 - p_param * b3

        if b2 == 0:
            break
        q_new = a0 / b2
        p_new = (a1 * b2 - a0 * b3) / (b2 ** 2)

        alpha_new = -p_new / 2.0
        discriminant = q_new - alpha_new ** 2
        beta_new = np.sqrt(discriminant) if discriminant >= 0 else 0

        # Перевірка критерію точності для комплексної пари
        if abs(alpha_new - alpha_val) < eps and abs(beta_new - beta_val) < eps:
            return alpha_new, beta_new, i

        alpha_val, beta_val = alpha_new, beta_new
    return alpha_val, beta_val, 100


if __name__ == "__main__":

    # ТАБУЛЯЦІЯ ТА ГРАФІК ТРАНСЦЕНДЕНТНОЇ ФУНКЦІЇ
    x_tabulation = np.arange(0, 12.6, 0.1)
    with open("tabulation.txt", "w", encoding="utf-8") as f:
        f.write("x\tF(x)\n")
        for x_item in x_tabulation:
            f.write(f"{x_item:.1f}\t{f_transcendent(x_item):.6f}\n")

    print("✓ Табуляцію трансцендентної функції збережено у 'tabulation.txt'.")

    # Малювання Графіка 1
    x_plot_line = np.linspace(0, 12.6, 500)
    plt.figure(figsize=(10, 5))
    plt.plot(x_plot_line, f_transcendent(x_plot_line), label="F(x) = sin(x) - 0.5", color='#4B0082', linewidth=2)
    plt.axhline(0, color='#4B0082', linewidth=0.8, alpha=0.7)

    root_up = np.arcsin(0.5)
    root_down = np.pi - np.arcsin(0.5)
    plt.scatter([root_up], [0], color='#00FF7F', s=90, zorder=5, edgecolors='black', label="Корінь при зростанні")
    plt.scatter([root_down], [0], color='#FF4500', s=90, zorder=5, edgecolors='black', label="Корінь при спаданні")

    plt.title("График трансцендентної функції", fontsize=12)
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()

    # ГРАФІКИ ЗБІЖНОСТІ ДЛЯ ТРАНСЦЕНДЕНТНОЇ ЧАСТИНИ
    plot_convergence("Збіжність методів: зростання функції", [0.1, 0.2, 0.3])
    plot_convergence("Збіжність методів: спадання функції", [3.1, 3.0, 2.9])

    # АЛГЕБРАЇЧНІ ОБЧИСЛЕННЯ ТА ГРАФІК 4 (УНІКАЛЬНА ФУНКЦІЯ)
    with open("coeffs.txt", "w", encoding="utf-8") as f:
        f.write("1.0 -3.0 4.0 -12.0")

    with open("coeffs.txt", "r", encoding="utf-8") as f:
        alg_coeffs = [float(val) for val in f.read().split()]

    real_root, r_iter = solve_real_horner(alg_coeffs, x0=4.0)
    c_alpha, c_beta, c_iter = solve_complex_lin(alg_coeffs, alpha0=1.0, beta0=1.0)

    print("\nЗВІТ ПО АЛГЕБРАЇЧНОМУ МОДУЛЮ")
    print(f"Дійсний корінь (Ньютон+Горнер): x = {real_root:.6f} (Ітерацій: {r_iter})")
    print(f"Комплексні корені (Метод Ліна):  x = {c_alpha:.6f} ± {c_beta:.6f}i (Ітерацій: {c_iter})")

    # Малювання Графіка 4
    x_vals_graph = np.linspace(-2, 6, 500)
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals_graph, p_algebraic(alg_coeffs, x_vals_graph), label="P(x) = x^3 - 3x^2 + 4x - 12", color='#4169E1',
             linewidth=2)
    plt.axhline(0, color='#4169E1', linewidth=0.8, alpha=0.7)
    plt.scatter([real_root], [0], color='#FFD700', s=100, zorder=5, edgecolors='#4169E1', label="Дійсний корінь")

    plt.title("Графік алгебраїчного многочлена (Унікальний)", fontsize=12)
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.xlim(-2.5, 6.5)
    plt.ylim(-60, 120)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()