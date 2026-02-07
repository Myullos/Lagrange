from scipy.interpolate import lagrange
import numpy as np
import matplotlib.pyplot as plt

# Compare Lagrange interpolation using Chebyshev nodes with the original function on a bounded interval.
# This illustrates how Chebyshev node placement reduces oscillations and improves approximation accuracy
# compared to equally spaced nodes.

def original_function(x):
    return 1 / (1 + x**2)

# Number of interpolation nodes
n = 10

# Generate Chebyshev nodes on [-1, 1] and scale them to [-5, 5]
k = np.arange(n + 1)
x_cheb = np.cos((2 * k + 1) * np.pi / (2 * (n + 1)))
x = 5 * x_cheb
y = original_function(x)

# Construct the Lagrange interpolating polynomial
f_Lag = lagrange(x, y)

# Dense sampling for visualization
xnew = np.linspace(-10, 10, 500)

# Plot interpolation vs original function
plt.plot(x, y, 'o', label='Chebyshev nodes')
plt.plot(xnew, f_Lag(xnew), label='Lagrange interpolation (Chebyshev nodes)')
plt.plot(xnew, original_function(xnew), '--', label='Original function')

plt.legend()
plt.xlim([-6, 6])
plt.ylim([-0.5, 2])
plt.grid(True)
plt.show()

