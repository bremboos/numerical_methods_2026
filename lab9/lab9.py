import numpy as np
import matplotlib.pyplot as plt


def non_linear_system_phi(X):
    """Цільова функція системи рівнянь (сума квадратів нев'язок кола та прямої)"""
    f1 = X[0] ** 2 + X[1] ** 2 - 4
    f2 = X[0] - X[1]
    return f1 ** 2 + f2 ** 2


def f1_rosenbrock(X):
    """Тестова функція Розенброка"""
    return 100 * (X[0] ** 2 - X[1]) ** 2 + (X[0] - 1) ** 2


def f2_power(X):
    """Тестова Степенева функція"""
    return (10 * (X[0] - X[1]) ** 2 + (X[0] - 1) ** 2) ** 4


def f3_root(X):
    """Тестова Коренева функція"""
    return (10 * (X[0] - X[1]) ** 2 + (X[0] - 1) ** 2) ** 0.25


def f4_wood(X):
    """Тестова функція Вуда"""
    return (100 * (X[1] - X[0] ** 2) ** 2 + (1 - X[0]) ** 2 +
            90 * (X[3] - X[2] ** 2) ** 2 + (1 - X[2]) ** 2 +
            10.1 * ((X[1] - 1) ** 2 + (X[3] - 1) ** 2) +
            19.8 * (X[1] - 1) * (X[3] - 1))


def f5_powell(X):
    """Тестова функція Пауелла"""
    return ((X[0] + 10 * X[1]) ** 2 + 5 * (X[2] - X[3]) ** 2 +
            10 * (X[0] - X[3]) ** 4 + (X[1] - 2 * X[2]) ** 4)


def f6_miele(X):
    """Тестова функція Мієлє"""
    return ((np.exp(X[0]) - X[1]) ** 4 + 100 * (X[1] - X[2]) ** 6 +
            np.tan(X[2] - X[3]) ** 4 + X[0] ** 8 + (X[3] - 1) ** 2)


def hooke_jeeves(func, X0, delta_start=0.2, q=2.0, p_param=1.0, eps=1e-5, max_iter=2000):
    """Багатовимірна оптимізація методом Хука-Дживса (Досліджуючий пошук та пошук по зразку)"""
    X_base = np.array(X0, dtype=float)
    n = len(X_base)
    delta = np.full(n, delta_start)

    trajectory = [np.copy(X_base)]
    steps = 0

    while np.max(delta) >= eps and steps < max_iter:
        steps += 1
        X_curr = np.copy(X_base)

        for i in range(n):
            f_old = func(X_curr)
            X_curr[i] += delta[i]
            if func(X_curr) < f_old: continue
            X_curr[i] -= 2 * delta[i]
            if func(X_curr) < f_old: continue
            X_curr[i] += delta[i]

        if func(X_curr) < func(X_base):
            X_pattern = X_curr + p_param * (X_curr - X_base)
            X_sample = np.copy(X_pattern)
            for i in range(n):
                f_s = func(X_sample)
                X_sample[i] += delta[i]
                if func(X_sample) < f_s: continue
                X_sample[i] -= 2 * delta[i]
                if func(X_sample) < f_s: continue
                X_sample[i] += delta[i]

            if func(X_sample) < func(X_curr):
                X_base = np.copy(X_sample)
            else:
                X_base = np.copy(X_curr)

            if not np.array_equal(trajectory[-1], X_base):
                trajectory.append(np.copy(X_base))
        else:
            delta /= q

    return np.array(trajectory), steps


test_functions = [
    {"name": "Розенброка", "func": f1_rosenbrock, "X0": [-1.2, 1.0], "delta": 0.1},
    {"name": "Степенева", "func": f2_power, "X0": [-1.2, 0.0], "delta": 0.1},
    {"name": "Коренева", "func": f3_root, "X0": [-1.2, 0.0], "delta": 0.1},
    {"name": "Вуда", "func": f4_wood, "X0": [-3.0, -1.0, -3.0, -1.0], "delta": 0.2},
    {"name": "Пауелла", "func": f5_powell, "X0": [-3.0, -1.0, 0.0, 1.0], "delta": 0.2},
    {"name": "Мієлє", "func": f6_miele, "X0": [1.0, 2.0, 2.0, 2.0], "delta": 0.2}
]

print("=== РЕЗУЛЬТАТИ ТЕСТУВАННЯ ШЕСТИ ФУНКЦІЙ ===")
for tf in test_functions:
    traj, steps = hooke_jeeves(tf["func"], tf["X0"], delta_start=tf["delta"], eps=1e-5)
    final_point = np.round(traj[-1], 4)
    final_val = tf["func"](traj[-1])
    print(f"{tf['name']}: x* = {list(final_point)}, Φ(x*) = {final_val:.2e}, кроків: {steps}")

print("\n=== РОЗВ'ЯЗОК ЗАДАНОЇ СИСТЕМИ РІВНЯНЬ ===")
traj_sys, steps_sys = hooke_jeeves(non_linear_system_phi, [-1.2, 0.0], delta_start=0.2, eps=1e-4)
print(f"Знайдено розв'язок системи: {traj_sys[-1]}")
print(f"Кількість кроків алгоритму: {steps_sys}")

with open("trajectory.txt", "w", encoding="utf-8") as f:
    f.write("Крок\tx1\tx2\tPhi(X)\n")
    for idx, pt in enumerate(traj_sys):
        f.write(f"{idx}\t{pt[0]:.6f}\t{pt[1]:.6f}\t{non_linear_system_phi(pt):.6e}\n")

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})

fig1, ax1 = plt.subplots(figsize=(7, 6))
x1, x2 = np.meshgrid(np.linspace(-2.0, 2.0, 300), np.linspace(-1.0, 3.0, 300))
Z_ros = 100 * (x1 ** 2 - x2) ** 2 + (x1 - 1) ** 2
ax1.contour(x1, x2, Z_ros, levels=np.logspace(-1, 3.5, 45), cmap='viridis', linewidths=0.8)
traj_ros, _ = hooke_jeeves(f1_rosenbrock, [-1.2, 1.0], delta_start=0.1)
ax1.plot(traj_ros[:, 0], traj_ros[:, 1], 'o-', color='#e74c3c', markersize=4, linewidth=1.2, label='Траєкторія')
ax1.set_title('Функція Розенброка та траєкторія спуску (Viridis)')
ax1.legend()

fig2, ax2 = plt.subplots(figsize=(6, 6))
theta = np.linspace(0, 2 * np.pi, 200)
ax2.plot(2 * np.cos(theta), 2 * np.sin(theta), label='$x_1^2 + x_2^2 - 4 = 0$', color='#2980b9')
x_l = np.linspace(-3, 3, 100)
ax2.plot(x_l, x_l, label='$x_1 - x_2 = 0$', color='#e67e22')
ax2.plot(traj_sys[:, 0], traj_sys[:, 1], 'o-', color='#2ecc71', markersize=5, linewidth=1.2, label='Траєкторія')
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_aspect('equal')
ax2.set_title('Графіки рівнянь системи та траєкторія спуску')
ax2.legend()

fig3, ax3 = plt.subplots(figsize=(7, 6))
xs1, xs2 = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
Z_phi = (xs1 ** 2 + xs2 ** 2 - 4) ** 2 + (xs1 - xs2) ** 2
ax3.contour(xs1, xs2, Z_phi, levels=45, cmap='plasma', linewidths=0.8)
ax3.plot(traj_sys[:, 0], traj_sys[:, 1], 'o-', color='#00decb', markersize=4, linewidth=1.2, label='Траєкторія')
ax3.set_xlim(-3, 3)
ax3.set_ylim(-3, 3)
ax3.set_aspect('equal')
ax3.set_title(r'Цільова функція $\Phi(X)$ та траєкторія спуску (Plasma)')
ax3.legend()

plt.show()