import requests
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['toolbar'] = 'None'

url = ("https://api.open-elevation.com/api/v1/lookup?locations="
       "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
       "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
       "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
       "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
       "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
       "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
       "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106")

response = requests.get(url)
data = response.json()
results = data["results"]
n_points = len(results)

print(f"Кількість вузлів: {n_points}")
print("\nТабуляція вузлів:")
print(" ID | Latitude  | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])
distances = [0.0]

for i in range(1, n_points):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

distances = np.array(distances)


def cubic_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)

    alpha = np.zeros(n)
    beta = np.zeros(n)
    gamma = np.zeros(n)
    delta = np.zeros(n)

    for i in range(1, n):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    A = np.zeros(n)
    B = np.zeros(n)
    for i in range(1, n):
        denom = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denom

    c = np.zeros(n + 1)
    for i in range(n - 1, 0, -1):
        c[i] = A[i] * c[i + 1] + B[i]

    a = y[:-1]
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c, d


def eval_spline(x_val, x, a, b, c, d):
    for i in range(len(x) - 1):
        if x[i] <= x_val <= x[i + 1]:
            dx = x_val - x[i]
            return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
    return a[-1] + b[-1] * (x_val - x[-2]) + c[-1] * (x_val - x[-2]) ** 2 + d[-1] * (x_val - x[-2]) ** 3


a_full, b_full, c_full, d_full = cubic_spline(distances, elevations)

total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n_points))
x_dense = np.linspace(distances[0], distances[-1], 500)
y_dense_full = np.array([eval_spline(xv, distances, a_full, b_full, c_full, d_full) for xv in x_dense])
grad_full = np.gradient(y_dense_full, x_dense) * 100
max_grad = np.max(grad_full)

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(distances, elevations, color='#2ca02c', marker='o', linestyle='-', linewidth=2, label='GPS точки')
ax1.set_title('Профіль висоти маршруту: Заросляк — Говерла', fontweight='bold', fontsize=14)
ax1.set_xlabel('Кумулятивна відстань (м)', fontsize=12)
ax1.set_ylabel('Висота (м)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)


textstr = f"Дистанція: {distances[-1] / 1000:.1f} км\nНабір висоти: {total_ascent:.0f} м\nМакс. ухил: {max_grad:.1f}%"
props = dict(boxstyle='round', facecolor='#f0f9e8', alpha=0.9, edgecolor='green')
ax1.text(0.03, 0.95, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)
ax1.legend(loc='lower right')

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.set_title('Вплив кількості вузлів на точність сплайна', fontweight='bold', fontsize=14)
ax2.plot(x_dense, y_dense_full, color='#333333', linestyle='-', linewidth=2.5, label='21 вузол (Еталон)')

splines_data = {} 
styles = {10: ('r--', 1.5), 15: ('b-.', 1.5), 20: ('g:', 2.5)}

for n_nodes in [10, 15, 20]:
    idx = np.linspace(0, n_points - 1, n_nodes, dtype=int)
    x_sub = distances[idx]
    y_sub = elevations[idx]
    a_sub, b_sub, c_sub, d_sub = cubic_spline(x_sub, y_sub)
    y_dense_sub = np.array([eval_spline(xv, x_sub, a_sub, b_sub, c_sub, d_sub) for xv in x_dense])
    splines_data[n_nodes] = y_dense_sub

    style, lw = styles[n_nodes]
    ax2.plot(x_dense, y_dense_sub, style, linewidth=lw, label=f'{n_nodes} вузлів')

ax2.set_xlabel('Відстань (м)', fontsize=12)
ax2.set_ylabel('Висота (м)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

props2 = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
ax2.text(0.03, 0.95, 'Висновок: більше вузлів =\nточніше відтворення рельєфу',
         transform=ax2.transAxes, fontsize=11, verticalalignment='top', bbox=props2)
ax2.legend(loc='lower right', shadow=True)

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.set_title('Абсолютна похибка інтерполяції', fontweight='bold', fontsize=14)

colors = {10: 'tab:red', 15: 'mediumblue', 20: 'forestgreen'}
for n_nodes in [10, 15, 20]:
    error = np.abs(y_dense_full - splines_data[n_nodes])
    max_err = np.max(error)
    ax3.plot(x_dense, error, color=colors[n_nodes], linestyle='-', label=f'{n_nodes} вузлів (Макс: {max_err:.1f} м)')

ax3.set_xlabel('Відстань (м)', fontsize=12)
ax3.set_ylabel('Похибка (м)', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(loc='upper right')

plt.show()

print("\n--- Додаткові характеристики маршруту ---")
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")

total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n_points))
print(f"Сумарний спуск (м): {total_descent:.2f}")
print(f"Максимальний підйом (%): {max_grad:.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

mass = 80
g = 9.81
energy = mass * g * total_ascent
print(f"Механічна робота (Дж): {energy:.2f}")
print(f"Енергія (ккал): {energy / 4184:.2f}")
