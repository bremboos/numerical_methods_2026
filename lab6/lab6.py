import numpy as np
import matplotlib.pyplot as plt


# генерація даних
def generate_data(n=100, x_val=2.5):
    A = np.random.uniform(1, 100, (n, n))
    x_true = np.full(n, x_val)
    b = np.dot(A, x_true)

    np.savetxt('matrix_A.txt', A)
    np.savetxt('vector_B.txt', b)
    print(f"Дані згенеровані та збережені (n={n}).")
    return n


# lu-розклад
def get_lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.eye(n)

    for k in range(n):
        for i in range(k, n):
            L[i, k] = A[i, k] - np.sum(L[i, :k] * U[:k, k])
        for i in range(k + 1, n):
            U[k, i] = (A[k, i] - np.sum(L[k, :k] * U[:k, i])) / L[k, k]

    return L, U


# розв'язок слар
def solve_lu(L, U, b):
    n = len(L)

    # пряма підстановка (lz = b)
    z = np.zeros(n)
    for k in range(n):
        z[k] = (b[k] - np.sum(L[k, :k] * z[:k])) / L[k, k]

    # зворотна підстановка (ux = z)
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = z[k] - np.sum(U[k, k + 1:] * x[k + 1:])
    return x


# основний цикл
n = generate_data(100)
A = np.loadtxt('matrix_A.txt')
b = np.loadtxt('vector_B.txt')

L, U = get_lu_decomposition(A)
x_0 = solve_lu(L, U, b)


def get_error(A, x, b):
    return np.max(np.abs(np.dot(A, x) - b))


eps_initial = get_error(A, x_0, b)
print(f"Початкова точність (нев'язка): {eps_initial:.2e}")

# ітераційне уточнення
eps_target = 1e-14
x_current = x_0.copy()
iteration = 0

error_history = [eps_initial]
iteration_history = [0]

print("\nЗапуск ітераційного уточнення...")
while iteration < 20:
    R = b - np.dot(A, x_current)

    delta_x = solve_lu(L, U, R)
    x_current = x_current + delta_x
    iteration += 1

    current_err = get_error(A, x_current, b)
    error_history.append(current_err)
    iteration_history.append(iteration)

    print(f"Ітерація {iteration}: помилка = {current_err:.2e}")

    if current_err <= eps_target:
        print("Досягнуто задану точність!")
        break

print(f"\nКінцевий результат досягнуто за {iteration} ітерацій.")

# побудова графіка збіжності
plt.figure(figsize=(10, 6))
plt.plot(iteration_history, error_history, marker='o', linestyle='-', color='b', linewidth=2)
plt.yscale('log')

plt.title('Залежність похибки розв\'язку від номера ітерації', fontsize=14)
plt.xlabel('Номер ітерації', fontsize=12)
plt.ylabel('Максимальна похибка (логарифмічна шкала)', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.7)

for i, txt in enumerate(error_history):
    plt.annotate(f"{txt:.1e}", (iteration_history[i], error_history[i]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

# візуалізація матриць
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im1 = axes[0].imshow(A, cmap='viridis', aspect='auto')
axes[0].set_title('Вихідна матриця A', fontsize=12)
axes[0].set_xlabel('Стовпці')
axes[0].set_ylabel('Рядки')
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(L, cmap='plasma', aspect='auto')
axes[1].set_title('Нижня трикутна матриця L', fontsize=12)
axes[1].set_xlabel('Стовпці')
fig.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(U, cmap='plasma', aspect='auto')
axes[2].set_title('Верхня трикутна матриця U', fontsize=12)
axes[2].set_xlabel('Стовпці')
fig.colorbar(im3, ax=axes[2])

plt.suptitle('Візуальне підтвердження правильності LU-розкладу', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()