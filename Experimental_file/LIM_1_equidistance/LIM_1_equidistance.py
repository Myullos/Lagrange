from scipy.interpolate import lagrange
import numpy as np
import matplotlib.pyplot as plt

# Compare Lagrange interpolation using equally spaced nodes.
# with the original function on a bounded interval.
# This demonstrates interpolation accuracy and oscillation behavior.
# (e.g. Runge phenomenon) for high-degree polynomials.

# Original function
def original_function(x):
    return 1 / (1 + x**2)

# Number of nodes (degree = n)
n = 10

# Equally spaced nodes in [-5, 5]
x = np.linspace(-5, 5, n + 1)
y = original_function(x)

# Lagrange interpolation
f_Lag = lagrange(x, y)

# Dense grid for plotting
xnew = np.linspace(-10, 10, 500)

# Plot
plt.plot(x, y, 'o', label='Equally spaced nodes')
plt.plot(xnew, f_Lag(xnew), label='Lagrange interpolation')
plt.plot(xnew, original_function(xnew), '--', label='Original function')

plt.legend()
plt.xlim([-6, 6])
plt.ylim([-0.5, 2])
plt.grid(True)
plt.show()

