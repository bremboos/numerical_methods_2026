import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sys


sys.setrecursionlimit(2000)

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


a, b = 0, 24

I0, _ = quad(f, a, b, epsabs=1e-13, epsrel=1e-13)
print("==========================================")
print(f"1. Точне значення інтегралу I0 = {I0:.12f}")
print("==========================================")

def simpson_composite(f, a, b, N):
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    I = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return I

N_values = np.arange(10, 1002, 2)
errors_simpson = []
N_opt = None
eps_opt = None

for N in N_values:
    I_N = simpson_composite(f, a, b, N)
    err = abs(I_N - I0)
    errors_simpson.append(err)
    if N_opt is None and err <= 1e-12:
        N_opt = N
        eps_opt = err

print(f"\n2. Задана точність 1e-12 досягається при N_opt = {N_opt}")
print(f"   Похибка при N_opt: eps_opt = {eps_opt:.2e}")

N0_approx = N_opt // 10
N0 = N0_approx + (8 - N0_approx % 8) if N0_approx % 8 != 0 else N0_approx
if N0 == 0: N0 = 8

I_N0 = simpson_composite(f, a, b, N0)
eps0 = abs(I_N0 - I0)
print(f"\n3. Вибране N0 (кратне 8) = {N0}")
print(f"   Значення інтегралу при N0: {I_N0:.12f}")
print(f"   Похибка eps0 = {eps0:.2e}")


I_N0_2_R = simpson_composite(f, a, b, N0 // 2)
I_R = I_N0 + (I_N0 - I_N0_2_R) / 15
epsR = abs(I_R - I0)
print(f"\n4. Метод Рунге-Ромберга:")
print(f"   Уточнене значення I_R = {I_R:.12f}")
print(f"   Похибка epsR = {epsR:.2e}")


I_N0_2 = simpson_composite(f, a, b, N0 // 2)
I_N0_4 = simpson_composite(f, a, b, N0 // 4)

den = 2 * I_N0_2 - (I_N0 + I_N0_4)
if abs(den) > 1e-16:
    I_E = (I_N0_2 ** 2 - I_N0 * I_N0_4) / den
else:
    I_E = I_N0

ratio = abs((I_N0_4 - I_N0_2) / (I_N0_2 - I_N0))
p = np.log(ratio) / np.log(2) if ratio > 0 else 0
epsE = abs(I_E - I0)

print(f"\n5. Метод Ейткена:")
print(f"   Уточнене значення I_E = {I_E:.12f}")
print(f"   Похибка epsE = {epsE:.2e}")
print(f"   Оцінка порядку методу p = {p:.4f}")

def adaptive_simpson(f, a, b, delta, counter):
    c = (a + b) / 2
    h = b - a

    y0, y_half, y1 = f(a), f(c), f(b)
    I1 = (h / 6) * (y0 + 4 * y_half + y1)
    counter[0] += 3

    c1 = (a + c) / 2
    c2 = (c + b) / 2
    y_14, y_34 = f(c1), f(c2)
    counter[0] += 2

    I2 = (h / 12) * (y0 + 4 * y_14 + y_half) + (h / 12) * (y_half + 4 * y_34 + y1)

    if abs(I1 - I2) <= delta:
        return I2
    else:
        return adaptive_simpson(f, a, c, delta / 2, counter) + \
            adaptive_simpson(f, c, b, delta / 2, counter)


print(f"\n6. Дослідження адаптивного алгоритму:")
deltas = np.logspace(-1, -7, 7)
errors_adapt = []
calls_adapt = []

for d in deltas:
    counter = [0]
    I_adapt = adaptive_simpson(f, a, b, d, counter)
    err_adapt = abs(I_adapt - I0)
    errors_adapt.append(err_adapt)
    calls_adapt.append(counter[0])
    print(f"   delta = {d:.0e} | Похибка = {err_adapt:.2e} | Викликів = {counter[0]}")


# --- Графік 1: Залежність точності Сімпсона від N ---
plt.figure(figsize=(10, 6))
plt.plot(N_values, errors_simpson, label=r'$\epsilon(N) = |I(N) - I_0|$', color='#800080', linewidth=2)
plt.axhline(1e-12, color='#FF8C00', linestyle='--', linewidth=2, label=r'Задана точність $10^{-12}$')

if N_opt:
    plt.scatter([N_opt], [eps_opt], color='#FF8C00', s=60, zorder=5)
    plt.text(N_opt + 15, eps_opt * 1.5, f'N={N_opt}', color='#FF8C00', fontweight='bold', fontsize=11)

plt.yscale('log')
plt.xlabel('Число розбиття відрізку, N')
plt.ylabel('Абсолютна похибка (логарифмічна шкала)')
plt.title('Залежність точності складової формули Сімпсона від N')
plt.grid(True, which="both", ls="--", alpha=0.5, color='gray')
plt.legend()
plt.show()  # Відображаємо перший графік. Після його закриття з'явиться другий.

# --- Графік 2: Аналіз адаптивного алгоритму ---
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = '#800080'
# Виправлено: додано r перед рядками для уникнення SyntaxWarning
ax1.set_xlabel(r'Параметр $\delta$ (логарифмічна шкала, зліва направо - зменшення $\delta$)')
ax1.set_ylabel(r'Абсолютна похибка $\epsilon$', color=color1, fontweight='bold')
line1 = ax1.plot(deltas, errors_adapt, marker='o', color=color1, linewidth=2, label=r'Похибка $\epsilon$')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.invert_xaxis()
ax1.grid(True, which="both", ls="--", alpha=0.5, color='gray')

ax2 = ax1.twinx()
color2 = '#FF8C00'
ax2.set_ylabel('Кількість обчислень функції f(x)', color=color2, fontweight='bold')
line2 = ax2.plot(deltas, calls_adapt, marker='s', color=color2, linestyle='--', linewidth=2, label='Обчислення f(x)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_yscale('log')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center')

plt.title(r'Аналіз адаптивного алгоритму ($\epsilon$ та виклики від $\delta$)')
fig.tight_layout()
plt.show()