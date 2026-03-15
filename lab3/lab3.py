import numpy as np
import matplotlib.pyplot as plt
import csv

def read_data(filename):
    x = []
    y = []
    try:
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                x.append(float(row['Month']))
                y.append(float(row['Temp']))
    except (FileNotFoundError, KeyError):
        x = list(range(1, 25))
        y = [-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0, -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3]
    return np.array(x), np.array(y)

#Формування системи для МНК
def form_system(x, y, m):
    size = m + 1
    B = np.zeros((size, size))
    C = np.zeros(size)
    for k in range(size):
        for l in range(size):
            B[k, l] = np.sum(x**(k + l))
        C[k] = np.sum(y * (x**k))
    return B, C

#Метод Гаусса з вибором головного елемента
def gauss_solve(A, b):
    n = len(b)
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    for k in range(n):
        max_row = k + np.argmax(np.abs(A[k:, k]))
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.dot(A[i, i+1:], x_sol[i+1:])) / A[i, i]
    return x_sol

def polynomial(x, coef):
    y_poly = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coef):
        y_poly += c * (x**i)
    return y_poly

def calculate_variance(y_true, y_approx):
    n = len(y_true)
    return np.sqrt(np.sum((y_approx - y_true)**2) / n) # [cite: 30]

x_data, y_data = read_data('temp_data.csv')
max_degree = 10
variances = []
coefficients = {}

print("Результати:")
print(f"{'m':<5} | {'Дисперсія δ':<15}")
print("-" * 25)

for m in range(1, max_degree + 1):
    B, C = form_system(x_data, y_data, m)
    coef = gauss_solve(B, C)
    y_approx = polynomial(x_data, coef)
    var = calculate_variance(y_data, y_approx)
    variances.append(var)
    coefficients[m] = coef
    print(f"{m:<5} | {var:<15.4f}")

optimal_m = np.argmin(variances) + 1
opt_coef = coefficients[optimal_m]
x_future = np.array([25, 26, 27])
y_future = polynomial(x_future, opt_coef)

print(f"\nОптимальний ступінь: {optimal_m}")
print(f"Прогноз на 25-27 місяці: {np.round(y_future, 2)}") # [cite: 192]

# Вікно 1: Апроксимація
plt.figure(1, figsize=(10, 6))
plt.scatter(x_data, y_data, color='red', label='Дані')
x_smooth = np.linspace(1, 24, 200)
plt.plot(x_smooth, polynomial(x_smooth, opt_coef), 'b-', label=f'МНК (m={optimal_m})')
plt.title('Графік апроксимації фактичних даних')
plt.xlabel('Місяць'); plt.ylabel('Температура'); plt.legend(); plt.grid(True)

# Вікно 2: Похибка
plt.figure(2, figsize=(10, 6))
errors = np.abs(y_data - polynomial(x_data, opt_coef))
plt.bar(x_data, errors, color='orange', alpha=0.7, label='|f(x) - phi(x)|')
plt.plot(x_data, errors, 'r-o', markersize=4)
plt.title('Табулювання та графік похибки')
plt.xlabel('Місяць'); plt.ylabel('Похибка'); plt.legend(); plt.grid(True)

plt.show()