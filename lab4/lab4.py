import numpy as np
import matplotlib.pyplot as plt



# 1. Визначення функцій та початкових даних


def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)


def M_prime_analytical(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)


def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


t0 = 1.0
exact_val = M_prime_analytical(t0)

print("--- Крок 1: Аналітичне розв'язання ---")
print(f"Точне значення M'(1): {exact_val:.10f}\n")

h_values_all = np.logspace(-20, 3, num=300)
errors_all = []

for h in h_values_all:
    try:
        approx = central_difference(M, t0, h)
        error = abs(approx - exact_val)
        errors_all.append(error)
    except Exception:
        errors_all.append(np.nan)

min_error_idx = np.nanargmin(errors_all)
best_h = h_values_all[min_error_idx]
min_error = errors_all[min_error_idx]

print("--- Крок 2: Оптимальний крок ---")
print(f"Найкраща точність досягнута при h0 = {best_h:.2e}")
print(f"Мінімальна похибка R0 = {min_error:.2e}\n")



h_fixed = 1e-3
D_h = central_difference(M, t0, h_fixed)
D_2h = central_difference(M, t0, 2 * h_fixed)
D_4h = central_difference(M, t0, 4 * h_fixed)

R1 = abs(D_h - exact_val)

print(f"--- Кроки 3-5: Чисельне диференціювання (h = {h_fixed}) ---")
print(f"D(h):   {D_h:.10f}")
print(f"D(2h):  {D_2h:.10f}")
print(f"Похибка R1: {R1:.2e}\n")

# Метод Рунге-Ромберга
y_R = D_h + (D_h - D_2h) / 3
R2 = abs(y_R - exact_val)

print("--- Крок 6: Метод Рунге-Ромберга ---")
print(f"Уточнене значення y_R: {y_R:.10f}")
print(f"Похибка R2: {R2:.2e}")
print(f"Покращення точності в {R1 / R2:.2f} разів\n")

# Метод Ейткена
numerator = (D_2h ** 2) - (D_4h * D_h)
denominator = 2 * D_2h - (D_4h + D_h)
y_E = numerator / denominator

p_est = np.log2(abs((D_4h - D_2h) / (D_2h - D_h)))
R3 = abs(y_E - exact_val)

print("--- Крок 7: Метод Ейткена ---")
print(f"Уточнене значення y_E: {y_E:.10f}")
print(f"Оціночний порядок точності p: {p_est:.2f}")
print(f"Похибка R3: {R3:.2e}\n")

if y_R < 0:
    print("Висновок: Вологість зменшується (швидкість від'ємна). Потрібно планувати полив.")



plt.figure(figsize=(18, 6))

# --- Графік 1: M(t) та M'(t) ---
plt.subplot(1, 3, 1)
t_range = np.linspace(0, 20, 400)
plt.plot(t_range, M(t_range), label='$M(t)$ (Вологість)', color='#1f77b4', linewidth=2)
plt.plot(t_range, M_prime_analytical(t_range), label="$M'(t)$ (Швидкість)", linestyle='--', color='#ff7f0e',
         linewidth=2)
plt.title('Модель вологості та її похідна')
plt.xlabel('Час $t$')
plt.ylabel('Значення')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# --- Графік 2: Залежність похибки R(h) від кроку h (широкий діапазон) ---
plt.subplot(1, 3, 2)
plt.loglog(h_values_all, errors_all, label='Похибка $R(h)$', color='purple')
plt.scatter(best_h, min_error, color='red', zorder=5, label=f'Оптимальний $h_0 \\approx 10^{{{np.log10(best_h):.1f}}}$')
plt.title('Похибка від кроку $h$ (загальна)')
plt.xlabel('Крок $h$ (логарифмічний масштаб)')
plt.ylabel('Абсолютна похибка (логарифмічний масштаб)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.gca().invert_xaxis()  # Інвертуємо вісь X для наочності (зменшення h зліва направо)
plt.legend()

# --- Графік 3: Порівняння похибок Рунге-Ромберга та Ейткена ---
plt.subplot(1, 3, 3)

h_vals_comp = np.logspace(-6, 0, 300)
err_std, err_rr, err_aitken = [], [], []

for h in h_vals_comp:
    dh = central_difference(M, t0, h)
    d2h = central_difference(M, t0, 2 * h)
    d4h = central_difference(M, t0, 4 * h)

    err_std.append(abs(dh - exact_val))
    err_rr.append(abs(dh + (dh - d2h) / 3 - exact_val))

    den = 2 * d2h - (d4h + dh)
    if den == 0:
        err_aitken.append(np.nan)
    else:
        err_aitken.append(abs((d2h ** 2 - d4h * dh) / den - exact_val))

plt.loglog(h_vals_comp, err_std, label='Центральна різниця $O(h^2)$', color='#1f77b4', linewidth=2)
plt.loglog(h_vals_comp, err_rr, label='Рунге-Ромберг', color='#ff7f0e', linewidth=2)
plt.loglog(h_vals_comp, err_aitken, label='Метод Ейткена', color='#2ca02c', linestyle='--', linewidth=2)

plt.title('Порівняння методів уточнення')
plt.xlabel('Крок $h$ (логарифмічний масштаб)')
plt.ylabel('Абсолютна похибка')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.gca().invert_xaxis()
plt.legend()

plt.tight_layout()
plt.show()