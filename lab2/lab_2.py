import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import math

# Обчислення розділених різниць
def get_divided_differences(x, y):
    n = len(y)
    table = np.zeros([n, n])
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])
    return table[0, :]


# Обчислення значення многочлена Ньютона
def newton_interpolation(x_nodes, coefs, x_val):
    n = len(coefs) - 1
    p = coefs[n]
    for k in range(1, n + 1):
        p = coefs[n - k] + (x_val - x_nodes[n - k]) * p
    return p


# Обчислення скінченних різниць
def get_finite_differences(y):
    n = len(y)
    table = np.zeros([n, n])
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i + 1, j - 1] - table[i, j - 1]
    return table[0, :]


# Інтерполяція факторіальними многочленами
def factorial_interpolation(x_nodes, y_nodes, x_val):
    h = x_nodes[1] - x_nodes[0]
    t = (x_val - x_nodes[0]) / h

    diffs = get_finite_differences(y_nodes)
    n = len(diffs)

    result = diffs[0]
    t_factorial = 1.0

    for k in range(1, n):
        t_factorial *= (t - k + 1)
        term = (diffs[k] * t_factorial) / math.factorial(k)
        result += term

    return result


# Функція для табуляції теоретичної моделі
def theoretical_model(x):
    return 0.00032 * x ** 2 + 0.00015 * x + 2.6


filename = 'data_lab2.csv'
if not os.path.exists(filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 't'])
        writer.writerows([[1000, 3], [2000, 5], [4000, 11], [8000, 28], [16000, 85]])

x_orig, y_orig = [], []
with open(filename, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        x_orig.append(float(row['n']))
        y_orig.append(float(row['t']))
x_orig = np.array(x_orig)
y_orig = np.array(y_orig)

#ДОСЛІДЖЕННЯ (5, 10, 20 ВУЗЛІВ)

node_counts = [5, 10, 20]
x_fine = np.linspace(1000, 16000, 500)
y_theory = theoretical_model(x_fine)

# Створюємо два окремих вікна
fig_newton = plt.figure("Метод Ньютона", figsize=(12, 10))
fig_fact = plt.figure("Факторіальні многочлени", figsize=(12, 10))

for idx, n in enumerate(node_counts):
    # Створюємо n рівновіддалених вузлів
    x_nodes = np.linspace(1000, 16000, n)
    y_nodes = theoretical_model(x_nodes) if n > 5 else y_orig

    # Обчислення для Ньютона
    coefs_newton = get_divided_differences(x_nodes, y_nodes)
    y_interp_newton = [newton_interpolation(x_nodes, coefs_newton, val) for val in x_fine]
    error_newton = np.abs(y_theory - np.array(y_interp_newton))

    # Обчислення для Факторіальних многочленів
    y_interp_fact = [factorial_interpolation(x_nodes, y_nodes, val) for val in x_fine]
    error_fact = np.abs(y_theory - np.array(y_interp_fact))

    #Вікно 1: графіки ньютона
    plt.figure(fig_newton.number)  # Перемикаємось на вікно Ньютона

    plt.subplot(len(node_counts), 2, 2 * idx + 1)
    plt.plot(x_fine, y_interp_newton, label=f'Newton n={n}', color='blue')
    plt.scatter(x_nodes, y_nodes, color='red', s=25, zorder=5)
    plt.title(f'Інтерполяція Ньютона ({n} вузлів)')
    plt.grid(True)
    plt.legend()

    plt.subplot(len(node_counts), 2, 2 * idx + 2)
    plt.plot(x_fine, error_newton, color='orange', label='Error ε(x)')
    plt.yscale('log')
    plt.title(f'Похибка Ньютона ({n} вузлів)')
    plt.grid(True)
    plt.legend()

    #Вікно 2: графіки факторіальних многочленів
    plt.figure(fig_fact.number)  # Перемикаємось на вікно Факторіальних многочленів

    plt.subplot(len(node_counts), 2, 2 * idx + 1)
    plt.plot(x_fine, y_interp_fact, label=f'Factorial n={n}', color='black')
    plt.scatter(x_nodes, y_nodes, color='red', s=25, zorder=5)
    plt.title(f'Факторіальна інтерполяція ({n} вузлів)')
    plt.grid(True)
    plt.legend()

    plt.subplot(len(node_counts), 2, 2 * idx + 2)
    plt.plot(x_fine, error_fact, color='green', label='Error ε(x)')
    plt.yscale('log')
    plt.title(f'Похибка факторіальна ({n} вузлів)')
    plt.grid(True)
    plt.legend()

# Оформлення відступів для обох вікон
plt.figure(fig_newton.number)
plt.tight_layout()

plt.figure(fig_fact.number)
plt.tight_layout()
plt.show()


print("--- Прогноз часу виконання для алгоритму при n = 6000 ---")

#Метод Ньютона
final_coefs = get_divided_differences(x_orig, y_orig)
prediction_newton = newton_interpolation(x_orig, final_coefs, 6000)
print(f"Метод Ньютона (на оригінальних даних): {prediction_newton:.2f} мс")

# Метод Факторіальних многочленів
x_equidistant = np.linspace(1000, 16000, 5)
y_equidistant = theoretical_model(x_equidistant)
prediction_fact = factorial_interpolation(x_equidistant, y_equidistant, 6000)
print(f"Факторіальні многочлени (на рівномірній сітці): {prediction_fact:.2f} мс")
print("-" * 65)