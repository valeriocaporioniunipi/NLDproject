import numpy as np
import matplotlib.pyplot as plt
from utils import rkf45_step, lorenz_system, estimate_correlation_dimension, integrate_system

# Parameters
t0 = 0.0
y0 = np.array([1.0, 1.0, 1.0])
t_max = 500.0
h = 0.01

# Integrate the system
t_values, y_values = integrate_system(lorenz_system, t0, y0, t_max, h)
# Plot the Lorenz attractor with matplotlib
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(y_values[:, 0], y_values[:, 1], y_values[:, 2], lw=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set initial view angle
ax.view_init(elev=18, azim=122) 
# Adjust the elevation (elev) and azimuth (azim) angles for the desired view

plt.tight_layout()
plt.show()

# Plot time series
fig, ax = plt.subplots(figsize=(10, 4))
y0_2 = np.array([1.0, 1.01, 0.99])
t_values_2, y_values_2 = integrate_system(lorenz_system, t0, y0_2, t_max, h)
# Plot time series of X coordinate
ax.plot(t_values, y_values[:, 0], color='blue', label='Orbit 1', fontsize = 20)
ax.plot(t_values_2, y_values_2[:, 0], color='black', alpha=0.5, label='Orbit 2', fontsize = 20)
ax.set_xlabel('t', fontsize = 20)
ax.set_ylabel('X', fontsize = 20)
ax.set_xlim(410, 450)

plt.tight_layout()
plt.savefig('lorenz_time_series_x.pdf')
plt.show()
log_epsilons,log_C_epsilon,correlation_dimension,intercept=estimate_correlation_dimension(y_values)

# # Plot log-log plot of epsilon vs C(epsilon)
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(log_epsilons, log_C_epsilon)
# ax.plot(log_epsilons,
#         correlation_dimension * log_epsilons + intercept,
#         color='red', label=f'd ={correlation_dimension:.2f})')
# ax.set_xlabel('log($\epsilon$)', fontsize = 20)
# ax.set_ylabel('log($C(\epsilon)$)')
# ax.set_title('Correlation Dimension of Lorenz Attractor', fontsize = 20)
# ax.legend(fontsize = 22)

# plt.tight_layout()
# plt.savefig('lorenz_fractal_dimension.pdf')
# plt.show()

