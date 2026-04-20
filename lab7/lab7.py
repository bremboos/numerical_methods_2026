import numpy as np
import matplotlib.pyplot as plt


def generate_and_save_data(n=100, filename_A="matrix_A.txt", filename_B="vector_B.txt"):
    # генерація випадкової матриці
    A = np.random.uniform(-10, 10, (n, n))

    # забезпечення строгого діагонального переважання
    for i in range(n):
        sum_row = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        A[i, i] = sum_row + np.random.uniform(1, 10)

    # точний розв'язок
    x_exact = np.full(n, 2.5)

    # вектор вільних членів
    B = np.dot(A, x_exact)

    # запис у файли
    np.savetxt(filename_A, A)
    np.savetxt(filename_B, B)

    return A, B, x_exact


# базові математичні функції
def read_data(filename_A="matrix_A.txt", filename_B="vector_B.txt"):
    A = np.loadtxt(filename_A)
    B = np.loadtxt(filename_B)
    return A, B


def multiply_matrix_vector(A, x):
    return np.dot(A, x)


def vector_norm(x):
    # максимальна норма вектора
    return np.max(np.abs(x))


def matrix_norm(A):
    # норма матриці
    return np.max(np.sum(np.abs(A), axis=1))


# ітераційні методи
def simple_iteration_method(A, b, eps=1e-14, max_iter=2000):
    n = len(b)
    x = np.ones(n)  # початкове наближення

    # параметр збіжності
    tau = 1.0 / matrix_norm(A)

    errors = []
    for k in range(max_iter):
        x_new = x - tau * (multiply_matrix_vector(A, x) - b)
        err = vector_norm(x_new - x)
        errors.append(err)

        if err < eps:
            return x_new, k + 1, errors
        x = x_new

    return x, max_iter, errors


def jacobi_method(A, b, eps=1e-14, max_iter=2000):
    n = len(b)
    x = np.ones(n)  # початкове наближення
    x_new = np.zeros(n)
    errors = []

    for k in range(max_iter):
        for i in range(n):
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - s) / A[i, i]

        err = vector_norm(x_new - x)
        errors.append(err)

        if err < eps:
            return x_new, k + 1, errors
        x = np.copy(x_new)

    return x, max_iter, errors


def seidel_method(A, b, eps=1e-14, max_iter=2000):
    n = len(b)
    x = np.ones(n)  # початкове наближення
    x_new = np.copy(x)
    errors = []

    for k in range(max_iter):
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        err = vector_norm(x_new - x)
        errors.append(err)

        if err < eps:
            return x_new, k + 1, errors
        x = np.copy(x_new)

    return x, max_iter, errors


# головний блок
if __name__ == "__main__":
    n = 100
    eps = 1e-14

    print(f"генеруємо матрицю {n}x{n} та вектор b...")
    generate_and_save_data(n)

    A, b = read_data()

    print(f"\nшукаємо розв'язок з точністю {eps}:")

    x_sim, iter_sim, err_sim = simple_iteration_method(A, b, eps)
    print(f"метод простої ітерації: {iter_sim} ітерацій.")

    x_jac, iter_jac, err_jac = jacobi_method(A, b, eps)
    print(f"метод якобі: {iter_jac} ітерацій.")

    x_sei, iter_sei, err_sei = seidel_method(A, b, eps)
    print(f"метод зейделя: {iter_sei} ітерацій.")

    # побудова графіка
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, iter_sim + 1), err_sim, label='метод простої ітерації', color='blue', linewidth=2)
    plt.plot(range(1, iter_jac + 1), err_jac, label='метод якобі', color='green', linewidth=2)
    plt.plot(range(1, iter_sei + 1), err_sei, label='метод зейделя', color='red', linewidth=2)

    # логарифмічна шкала
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('кількість ітерацій', fontsize=12)
    plt.ylabel('норма різниці $||x^{(k+1)} - x^{(k)}||$', fontsize=12)
    plt.title('порівняння збіжності ітераційних методів', fontsize=14)
    plt.legend(fontsize=12)

    plt.show()