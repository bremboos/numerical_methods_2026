import numpy as np
import matplotlib.pyplot as plt

# Цільові функції

def rosenbrock(X):
    """Тестова функція Розенброка"""
    return 100 * (X[0] ** 2 - X[1]) ** 2 + (X[0] - 1) ** 2


def non_linear_system_phi(X):
    """Цільова функція для системи нелінійних рівнянь"""
    f1 = X[0] ** 2 + X[1] ** 2 - 4
    f2 = X[0] - X[1]
    return f1 ** 2 + f2 ** 2


# алгоритм Хука-Дживса із захистом від зациклювання


def hooke_jeeves_safe(func, X0, delta_start=0.2, q=2.0, p_param=1.0, eps=1e-4, max_iter=1000):
    """
    Реалізація методу Хука-Дживса із захистом від нескінченних циклів.
    """
    X_base = np.array(X0, dtype=float)
    n = len(X_base)
    delta = np.full(n, delta_start)

    trajectory = [np.copy(X_base)]
    steps = 0

    while np.max(delta) >= eps and steps < max_iter:
        steps += 1
        # 1. Досліджуючий пошук
        X_curr = np.copy(X_base)

        for i in range(n):
            f_old = func(X_curr)

            # Крок у позитивному напрямку
            X_curr[i] += delta[i]
            if func(X_curr) < f_old:
                continue

            # Крок у негативному напрямку
            X_curr[i] -= 2 * delta[i]
            if func(X_curr) < f_old:
                continue

            # Якщо обидва кроки невдалі — повертаємо координату назад
            X_curr[i] += delta[i]

        # 2. Перевірка успішності досліджуючого пошуку
        if func(X_curr) < func(X_base):
            # Пошук по зразку
            X_pattern = X_curr + p_param * (X_curr - X_base)

            # Досліджуючий пошук навколо точки зразка
            X_sample = np.copy(X_pattern)
            for i in range(n):
                f_s = func(X_sample)
                X_sample[i] += delta[i]
                if func(X_sample) < f_s: continue
                X_sample[i] -= 2 * delta[i]
                if func(X_sample) < f_s: continue
                X_sample[i] += delta[i]

            # Якщо крок по зразку успішний
            if func(X_sample) < func(X_curr):
                X_base = np.copy(X_sample)
            else:
                X_base = np.copy(X_curr)

            # Запобігаємо дублюванню однакових точок у траєкторії
            if not np.array_equal(trajectory[-1], X_base):
                trajectory.append(np.copy(X_base))
        else:
            # Зменшуємо крок, якщо не знайшли покращення
            delta /= q

    return np.array(trajectory), steps


# 3. Обчислення траєкторій

# Розрахунок для функції Розенброка (старт з точки як на твоїх перших графіках)
traj_rosen, steps_rosen = hooke_jeeves_safe(rosenbrock, [-1.2, 1.0], delta_start=0.1, eps=1e-3)

# Розрахунок для системи нелінійних рівнянь
traj_sys, steps_sys = hooke_jeeves_safe(non_linear_system_phi, [-1.2, 0.0], delta_start=0.2, eps=1e-4)

# Запис точок траєкторії системи в файл
with open("trajectory.txt", "w", encoding="utf-8") as f:
    f.write("Крок\tx1\tx2\tPhi(X)\n")
    for idx, pt in enumerate(traj_sys):
        f.write(f"{idx}\t{pt[0]:.6f}\t{pt[1]:.6f}\t{non_linear_system_phi(pt):.6e}\n")

print(f"Функція Розенброка: пораховано успішно за {steps_rosen} ітерацій.")
print(f"Система рівнянь: пораховано успішно за {steps_sys} ітерацій.")

# 4. Візуалізація

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})

# --- Графік 1: Функція Розенброка ---
fig1, ax1 = plt.subplots(figsize=(7, 6))
x1, x2 = np.meshgrid(np.linspace(-2.0, 2.0, 300), np.linspace(-1.0, 3.0, 300))
Z_ros = 100 * (x1 ** 2 - x2) ** 2 + (x1 - 1) ** 2

levels_rosen = np.logspace(-1, 3.5, 45)
ax1.contour(x1, x2, Z_ros, levels=levels_rosen, cmap='viridis', linewidths=0.8)
ax1.plot(traj_rosen[:, 0], traj_rosen[:, 1], 'o-', color='#e74c3c', markersize=4, linewidth=1.2,
         label='Траєкторія спуску')
ax1.set_title('Функція Розенброка та траєкторія спуску (Viridis)')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.legend()

# --- Графік 2: Графіки рівнянь системи ---
fig2, ax2 = plt.subplots(figsize=(6, 6))
theta = np.linspace(0, 2 * np.pi, 200)

ax2.plot(2 * np.cos(theta), 2 * np.sin(theta), label='$x_1^2 + x_2^2 - 4 = 0$', color='#2980b9', linewidth=1.8)
x_l = np.linspace(-3, 3, 100)
ax2.plot(x_l, x_l, label='$x_1 - x_2 = 0$', color='#e67e22', linewidth=1.8)

ax2.plot(traj_sys[:, 0], traj_sys[:, 1], 'o-', color='#2ecc71', markersize=5, linewidth=1.2, label='Траєкторія спуску')
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_aspect('equal')
ax2.set_title('Графіки рівнянь системи та траєкторія спуску')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.legend()

# --- Графік 3: Цільова функція Phi(X) ---
fig3, ax3 = plt.subplots(figsize=(7, 6))
xs1, xs2 = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
Z_phi = (xs1 ** 2 + xs2 ** 2 - 4) ** 2 + (xs1 - xs2) ** 2

levels_phi = np.linspace(0, 50, 45)
ax3.contour(xs1, xs2, Z_phi, levels=levels_phi, cmap='plasma', linewidths=0.8)
ax3.plot(traj_sys[:, 0], traj_sys[:, 1], 'o-', color='#00decb', markersize=4, linewidth=1.2, label='Траєкторія спуску')
ax3.set_xlim(-3, 3)
ax3.set_ylim(-3, 3)
ax3.set_aspect('equal')
ax3.set_title(r'Цільова функція $\Phi(X)$ та траєкторія спуску (Plasma)')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.legend()

plt.show()
