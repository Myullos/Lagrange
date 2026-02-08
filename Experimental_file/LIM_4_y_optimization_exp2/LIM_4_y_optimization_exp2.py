from scipy.interpolate import lagrange
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

# Lagrange interpolation based on Chebyshev nodes,
# where the y-values of the nodes are iteratively perturbed
# to minimize the global squared error integral
# between the interpolant and the original function.

# Original target function
def original_function(x):
    return 1 / (1 + x**2)

# Squared error integral over the interval [-5, 5]
def error_integral(f_interp):
    func = lambda x: (original_function(x) - f_interp(x))**2
    return quad(func, -5, 5)[0]

# Number of interpolation nodes
n = 10

# Generate Chebyshev nodes on [-1, 1] and scale to [-5, 5]
x_cheb = np.cos((2*np.arange(n+1) + 1) / (2*(n+1)) * np.pi)
x = 5 * x_cheb
y = original_function(x)

# Initial Lagrange interpolant and error
f_Lag = lagrange(x, y)
best_error = error_integral(f_Lag)
best_y = y.copy()

# Optimization parameters
np.random.seed(0)
iterations = 1000
 # Random perturbation amplitude for y-values
step_size = 0.001

# Store error history
error_history = [best_error]

# Random search optimization of node y-values
for i in range(iterations):
    # Randomly perturb y-values
    trial_y = best_y + np.random.uniform(-step_size, step_size, size=len(y))
    f_trial = lagrange(x, trial_y)
    trial_error = error_integral(f_trial)

    # Accept update if error is reduced
    if trial_error < best_error:
        best_error = trial_error
        best_y = trial_y
        f_Lag = f_trial

    error_history.append(best_error)

# Visualization
xnew = np.linspace(-5, 5, 500)
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# --- Top: optimized interpolation ---
axes[0].plot(x, best_y, 'o', label='Optimized nodes (y adjusted)')
axes[0].plot(xnew, f_Lag(xnew), label='Optimized interpolation')
axes[0].plot(xnew, original_function(xnew), '--', label='Original function')
axes[0].legend()
axes[0].set_xlim([-6, 6])
axes[0].set_ylim([0, 1.4])
axes[0].set_title(f'Lagrange Interpolation Optimized (Final Error = {best_error:.5e})')
axes[0].grid(True)

# --- Bottom: error evolution ---
axes[1].plot(error_history, label='Squared error integral')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Error')
axes[1].set_title('Error Evolution During Optimization')
axes[1].set_yscale('log')
axes[1].grid(True, which='both')
axes[1].legend()

plt.tight_layout()
plt.show()

print(best_error)
