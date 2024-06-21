import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

# Define the Lorenz system
def lorenz_system(t, state, sigma=10.0, beta=8.0/3.0, rho=28.0):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
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

# Function to compute pairwise distances
def compute_pairwise_distances(y_values):
    distances = cdist(y_values, y_values)
    return distances[np.triu_indices(len(y_values), k=1)]

# Function to compute Nx(epsilon)
def compute_Nx(distances, epsilon):
    return np.sum(distances < epsilon)

# Function to estimate correlation dimension
def estimate_correlation_dimension(y_values, num_points=5000):
    # Subsample y_values to reduce computation time
    #subsample_indices = np.random.choice(len(y_values), num_points, replace=False)
    y_values_subsampled = y_values[-num_points:]

    # Compute pairwise distances
    distances = compute_pairwise_distances(y_values_subsampled)
    max_distance = np.max(distances)

    # Define a range of epsilon values
    epsilons = np.logspace(np.log10(0.01), np.log10(max_distance), num_points)

    # Compute Nx(epsilon) for each epsilon
    C_epsilon = np.array([compute_Nx(distances, epsilon) for epsilon in epsilons])

    # Filter out invalid values
    valid_indices = (C_epsilon > 0)
    log_epsilons = np.log(epsilons[valid_indices])
    log_C_epsilon = np.log(C_epsilon[valid_indices])
    valid_indices = (log_C_epsilon <= 11) & (log_C_epsilon >= 5)
    log_epsilons_valid = log_epsilons[valid_indices]
    log_C_epsilon_valid = log_C_epsilon[valid_indices]
    # Perform linear regression to estimate correlation dimension
    model = LinearRegression()
    model.fit(log_epsilons_valid.reshape(-1, 1), log_C_epsilon_valid)
    correlation_dimension = model.coef_[0]
    intercept = model.intercept_

    return log_epsilons, log_C_epsilon, correlation_dimension, intercept

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