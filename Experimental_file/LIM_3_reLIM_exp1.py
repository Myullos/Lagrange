from scipy.interpolate import lagrange
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

# Lagrange interpolation based on Chebyshev nodes,
# followed by local refinement on subintervals where
# the area error exceeds a prescribed threshold.
# Refinement is performed by constructing additional
# local Lagrange interpolants.

# Original target function
def original_function(x):
    return 1 / (1 + x**2)

# Helper function to construct a Lagrange interpolating polynomial
def create_lagrange(x, y):
    return lagrange(x, y)

# Number of interpolation nodes (degree = n)
n = 10

# Generate Chebyshev nodes on [-1, 1] and scale them to [-5, 5]
x_main = np.cos((2*np.arange(1, n+2) - 1) / (2*(n+1)) * np.pi)
x_main = 5 * x_main
sort_idx = np.argsort(x_main)
x_main = x_main[sort_idx]
y_main = original_function(x_main)

# Construct the main global Lagrange interpolant
f_main = create_lagrange(x_main, y_main)

# --- Refinement parameters ---
area_threshold = 1e-6     # Area error threshold for refinement
max_refine_points = 5     # Number of points for local re-interpolation

sub_interpolations = []
x_sub_points = []

# Evaluate interpolation error on each subinterval
for i in range(len(x_main) - 1):
    a, b = x_main[i], x_main[i + 1]
    if a >= b:
        a, b = b, a

    # Compute area error on [a, b] via numerical integration
    abs_diff = lambda t: np.abs(original_function(t) - f_main(t))
    area, _ = quad(abs_diff, a, b)

    # Perform local re-interpolation if area error exceeds threshold
    if area > area_threshold:
        x_sub = np.linspace(a, b, max_refine_points)
        y_sub = original_function(x_sub)
        f_sub = create_lagrange(x_sub, y_sub)
        sub_interpolations.append((a, b, f_sub))
        x_sub_points.extend(x_sub)

# --- Visualization ---
x_plot = np.linspace(-10, 10, 1000)
y_true = original_function(x_plot)
y_main_interp = f_main(x_plot)

plt.figure(figsize=(10, 6))

# Main Chebyshev interpolation nodes
plt.plot(x_main, y_main, 'ko', label='Main Chebyshev nodes')

# Additional points used for local refinement
if x_sub_points:
    x_sub_points = np.unique(x_sub_points)
    y_sub_values = original_function(x_sub_points)
    plt.plot(x_sub_points, y_sub_values, 'r.', markersize=10, label='Refined raw data')

# Original function and main interpolant
plt.plot(x_plot, y_true, '--', color='cyan', label='Original function', linewidth=6.5)

# Locally refined Lagrange interpolants
for idx, (a, b, f_sub) in enumerate(sub_interpolations):
    x_sub_plot = np.linspace(a, b, 200)
    y_sub_plot = f_sub(x_sub_plot)
    plt.plot(
        x_sub_plot,
        y_sub_plot,
        color='red',
        linewidth=1.5,
        label='Refined Lagrange' if idx == 0 else ""
    )

plt.legend()
plt.title('Lagrange Interpolation (Chebyshev Nodes) with Area-Based Refinement')
plt.xlim([-6, 6])
plt.ylim([0, 1.4])
plt.grid(True)
plt.show()

