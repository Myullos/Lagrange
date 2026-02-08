import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import lagrange

# Curvature-Driven Interpolation using Local Bubble Functions
# Starting from a piecewise linear interpolant:
#   1. Local "bubble" functions are added on each interval.
#   2. Coefficients of the bubbles are optimized to reduce
#      the squared curvature (second derivative squared).
#   3. Weak global alignment with the Lagrange interpolant
#      is added to stabilize the solution.

# Original function to approximate
def f_true(x):
    return 1 / (1 + x**2)  # Runge function, classic test case

# Domain and number of intervals
a, b = -5.0, 5.0
n = 10

# Chebyshev nodes for better interpolation stability
k = np.arange(n + 1)
xs = (a + b) / 2 + (b - a) / 2 * np.cos((2*k + 1) * np.pi / (2*(n + 1)))
xs = np.sort(xs)
ys = f_true(xs)

# Piecewise linear interpolant as initial approximation
def piecewise_linear(x):
    return np.interp(x, xs, ys)

P1 = piecewise_linear(xs)

# Global Lagrange interpolant (for weak global reference)
P_L = lagrange(xs, ys)
PL = P_L(xs)

# Local bubble function supported on each interval
def bubble(x, i):
    """Bubble function on interval [xs[i], xs[i+1]].
    Peaks at the center, zero at endpoints. Used to locally
    adjust curvature without affecting nodes directly."""
    xL, xR = xs[i], xs[i + 1]
    h = xR - xL
    t = (x - xL) / h
    out = np.zeros_like(x)
    mask = (x >= xL) & (x <= xR)
    out[mask] = t[mask] * (1 - t[mask])
    return out

# Dense evaluation grid
xx = np.linspace(a, b, 3000)
P1_xx = piecewise_linear(xx)
PL_xx = P_L(xx)

# Regularization parameter controlling alignment with Lagrange
lam = 1e-2

# Initialize bubble coefficients
alphas = np.zeros(n)

# Optimize bubble coefficients per interval
for i in range(n):
    phi = bubble(xx, i)

    def objective(alpha):
        # Add bubble scaled by alpha
        f = P1_xx + alpha * phi
        # Approximate second derivative
        d2 = np.gradient(np.gradient(f, xx), xx)
        # Squared curvature integral
        curvature = simpson(d2**2, xx)
        # Weak alignment with global Lagrange interpolant
        align = simpson((f - PL_xx)**2, xx)
        return curvature + lam * align

    # Simple grid search to find optimal alpha
    grid = np.linspace(-5, 5, 400)
    vals = np.array([objective(a_) for a_ in grid])
    alphas[i] = grid[np.argmin(vals)]

# Curvature-corrected interpolant
def curvature_lagrange_interp(x):
    y = piecewise_linear(x)
    for i in range(n):
        y += alphas[i] * bubble(x, i)
    return y

# Evaluate interpolants on dense grid
xx = np.linspace(a, b, 2000)
y_true = f_true(xx)
y_lin  = piecewise_linear(xx)
y_lag  = P_L(xx)
y_curv = curvature_lagrange_interp(xx)

# Plot results
plt.figure(figsize=(10,6))
plt.plot(xx, y_true, 'k', lw=2, label="True function")
plt.plot(xx, y_lag, ':', lw=2, label="Lagrange")
plt.plot(xx, y_curv, '-.', lw=2, label="Curvature-corrected")
plt.scatter(xs, ys, c='red', zorder=5, label="Nodes")
plt.grid()
plt.legend()
plt.show()

# Plot optimized bubble coefficients
plt.figure(figsize=(8,4))
plt.bar(range(n), alphas)
plt.grid()
plt.xlabel("Interval index i")
plt.ylabel("alpha_i")
plt.title("Optimized bubble coefficients")
plt.show()

# Compute integrated squared errors
def error_integral(f_approx):
    return simpson((y_true - f_approx)**2, xx)

err_lin  = error_integral(y_lin)
err_lag  = error_integral(y_lag)
err_curv = error_integral(y_curv)

print(f"Squared error (piecewise linear): {err_lin:.6f}")
print(f"Squared error (Lagrange):         {err_lag:.6f}")
print(f"Squared error (curvature-corrected): {err_curv:.6f}")

improvement = (err_lin - err_curv) / err_lin * 100
print(f"Improvement by curvature correction: {improvement:.2f}%")

# Optimized alpha values per interval
for i, a_i in enumerate(alphas):
    print(f"[{xs[i]: .3f}, {xs[i+1]: .3f}] : alpha = {a_i: .6f}")


