import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Rössler system
def rossler_system(t, state, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    dx_dt = -y - z
    dy_dt = x + a * y
    dz_dt = b + z * (x - c)
    return np.array([dx_dt, dy_dt, dz_dt])

# Implement the RKF45 method
def rkf45_step(f, t, y, h, atol=1e-6, rtol=1e-6):
    c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    a = [
        [0],
        [1/4],
        [3/32, 9/32],
        [1932/2197, -7200/2197, 7296/2197],
        [439/216, -8, 3680/513, -845/4104],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ]
    b = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    b_star = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])

    k = []
    for i in range(6):
        if i == 0:
            y_temp = y
        else:
            y_temp = y + h * sum(a[i][j] * k[j] for j in range(i))
        k.append(f(t + c[i] * h, y_temp))

    y_new = y + h * sum(b[i] * k[i] for i in range(6))
    y_star = y + h * sum(b_star[i] * k[i] for i in range(6))

    error = np.abs(y_new - y_star)
    max_error = np.max(error / (atol + rtol * np.abs(y_new)))
    if max_error == 0:
        h_new = h * 2
    else:
        h_new = h * min(2, max(0.1, 0.84 * (1 / max_error) ** (1/4)))

    return y_new, h_new, max_error

# Integrate the Rössler system
def integrate_system(f, t0, y0, t_max, h):
    t_values = [t0]
    y_values = [y0]
    t = t0
    y = y0

    while t < t_max:
        if t + h > t_max:
            h = t_max - t
        y, h, error = rkf45_step(f, t, y, h)
        t += h
        t_values.append(t)
        y_values.append(y)

    return np.array(t_values), np.array(y_values)

# Parameters
t0 = 0.0
y0 = np.array([1.0, 1.0, 1.0])
t_max = 2000.0
h = 0.01

# Integrate the system
t_values, y_values = integrate_system(rossler_system, t0, y0, t_max, h)

# Plot the Rössler attractor with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(y_values[:, 0], y_values[:, 1], y_values[:, 2], lw=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Rössler Attractor')

# Create time delay embedding
delay = 10
embedding_dim = 2
delay_coords = []

for i in range(len(y_values) - delay * (embedding_dim - 1)):
    point = []
    for j in range(embedding_dim):
        point.append(y_values[i + j * delay, 0]+y_values[i + j * delay, 1]+y_values[i + j * delay, 2])  # Use the first coordinate x(t)
    delay_coords.append(point)

delay_coords = np.array(delay_coords)

# Plot the time delay embedding in 2D
plt.figure()
plt.plot(delay_coords[:, 0], delay_coords[:, 1], lw=0.5)
plt.xlabel('X(t)')
plt.ylabel(f'X(t + {delay})')
plt.title('2D immersion of Rössler Attractor')
plt.show()
