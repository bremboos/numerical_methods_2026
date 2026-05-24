import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x + y

def exact_sol(x):
    return 2 * np.exp(x) - x - 1

def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h, y + h*k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def solve_rk4(f, a, b, y0, h):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(len(x) - 1):
        y[i+1] = rk4_step(f, x[i], y[i], h)
    return x, y

def rk4_auto_step(f, a, b, y0, tol, h0):
    x_vals = [a]
    y_vals = [y0]
    h_vals = [h0]
    h = h0
    x = a
    y = y0
    while x < b:
        if x + h > b:
            h = b - x
        y_h = rk4_step(f, x, y, h)
        y_half_1 = rk4_step(f, x, y, h/2)
        y_half_2 = rk4_step(f, x + h/2, y_half_1, h/2)
        error = (16/15) * abs(y_half_2 - y_h)
        if error <= tol:
            x += h
            y = y_half_2
            x_vals.append(x)
            y_vals.append(y)
            h_vals.append(h)
        if error > 0:
            h = h * (tol / error)**0.2
        else:
            h *= 2
    return np.array(x_vals), np.array(y_vals), np.array(h_vals)

def solve_adams_pc(f, a, b, y0, h):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    y_pr = np.zeros(len(x))
    y[0] = y0
    if len(x) > 1:
        y[1] = rk4_step(f, x[0], y[0], h)
        y_pr[0] = y[0]
        y_pr[1] = y[1]
    for i in range(1, len(x) - 1):
        fn = f(x[i], y[i])
        fn_1 = f(x[i-1], y[i-1])
        y_pred = y[i] + (h/2) * (3*fn - fn_1)
        y_pr[i+1] = y_pred
        fn_pred = f(x[i+1], y_pred)
        y[i+1] = y[i] + (h/2) * (fn_pred + fn)
    return x, y, y_pr

def adams_auto_step(f, a, b, y0, tol, h0):
    x_vals = [a]
    y_vals = [y0]
    h_vals = [h0]
    h = h0
    x = a
    y = y0
    y_prev = y0
    x_prev = a
    y1 = rk4_step(f, a, y0, h)
    x += h
    y = y1
    x_vals.append(x)
    y_vals.append(y)
    h_vals.append(h)
    while x < b:
        if x + h > b:
            h = b - x
        fn = f(x, y)
        fn_1 = f(x_prev, y_prev)
        y_pred = y + (h/2) * (3*fn - fn_1)
        fn_pred = f(x + h, y_pred)
        y_corr = y + (h/2) * (fn_pred + fn)
        error = abs(y_corr - y_pred) / 6
        if error <= tol:
            x_prev = x
            y_prev = y
            x += h
            y = y_corr
            x_vals.append(x)
            y_vals.append(y)
            h_vals.append(h)
        if error > 0:
            h = h * (tol / error)**0.5
        else:
            h *= 2
    return np.array(x_vals), np.array(y_vals), np.array(h_vals)

a, b = 0.0, 1.0
y0 = 1.0
h_adams = 0.1
h_rk4 = 0.01
tol = 1e-4

x_exact = np.linspace(a, b, 200)
y_exact = exact_sol(x_exact)

x_adams, y_adams, y_pr_adams = solve_adams_pc(f, a, b, y0, h_adams)
x_rk4, y_rk4 = solve_rk4(f, a, b, y0, h_rk4)

plt.figure(1, figsize=(10, 6))
plt.plot(x_exact, y_exact, label="Точний розв'язок", linewidth=2)
plt.plot(x_adams, y_adams, marker='o', label=f"Адамс 2-го порядку, h = {h_adams}")
plt.plot(x_rk4, y_rk4, label=f"Рунге-Кутта 4-го порядку, h = {h_rk4}", linewidth=2)
plt.title("Порівняння точного та чисельних розв'язків")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.5)
plt.legend()

err_adams_exact = y_adams - exact_sol(x_adams)
err_adams_eval = (y_adams - y_pr_adams) / 6 * (-1)
err_rk4_exact = y_rk4 - exact_sol(x_rk4)

x_rk4_half, y_rk4_half = solve_rk4(f, a, b, y0, h_rk4/2)
err_rk4_runge = np.zeros(len(x_rk4))
for i in range(len(x_rk4)):
    if 2*i < len(y_rk4_half):
        err_rk4_runge[i] = (16/15) * (y_rk4_half[2*i] - y_rk4[i])

plt.figure(2, figsize=(10, 6))
plt.plot(x_adams, err_adams_exact, marker='o', label="П.3. Адамс: y_n - y(x_n)")
plt.plot(x_adams, err_adams_eval, marker='s', label="П.4. Адамс: R_2^kor")
plt.plot(x_rk4, err_rk4_exact, label="П.7. РК4: y_n - y(x_n)")
plt.plot(x_rk4, err_rk4_runge, label="П.8. РК4: похибка за методом Рунге")
plt.axhline(0, color='gray', linewidth=0.5)
plt.title("Порівняння локальних похибок")
plt.xlabel("x")
plt.ylabel("похибка")
plt.grid(True, alpha=0.5)
plt.legend()

h_vals_test = [0.1, 0.05, 0.025, 0.0125, 0.00625]
max_errs = []
for ht in h_vals_test:
    xt, yt = solve_rk4(f, a, b, y0, ht)
    max_errs.append(np.max(np.abs(yt - exact_sol(xt))))

plt.figure(3, figsize=(10, 6))
plt.loglog(h_vals_test, max_errs, marker='o', linestyle='-', linewidth=2)
for i, ht in enumerate(h_vals_test):
    plt.text(ht, max_errs[i], f" h={ht}")
plt.title("П.7. Залежність максимальної похибки РК4 від кроку h")
plt.xlabel("h")
plt.ylabel("max |y_n - y(x_n)|")
plt.grid(True, which="both", ls="--", alpha=0.5)

x_adams_auto, y_adams_auto, h_adams_auto = adams_auto_step(f, a, b, y0, tol, 0.1)
x_rk4_auto, y_rk4_auto, h_rk4_auto = rk4_auto_step(f, a, b, y0, tol, 0.01)

plt.figure(4, figsize=(10, 6))
plt.step(x_adams_auto, h_adams_auto, where='post', label="П.5. Адамс 2-го порядку", linewidth=2)
plt.step(x_rk4_auto, h_rk4_auto, where='post', label="П.9. РК4", linewidth=2)
plt.title("Порівняння автоматичного вибору кроку")
plt.xlabel("x")
plt.ylabel("h")
plt.grid(True, alpha=0.5)
plt.legend()

plt.show()