import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import lagrange

# Curvature-driven interpolation with local bubble functions.
# Starting from a piecewise linear interpolant, local bubble corrections
# are introduced and their coefficients are optimized to minimize
# the integrated squared curvature.
# A weak global shape reference is imposed solely for stabilization.

def f_true(x):
    return 1 / (1 + x**2)

a, b = -5.0, 5.0
n = 10

# Chebyshev nodes on [a, b]
k = np.arange(n + 1)
xs = (a + b) / 2 + (b - a) / 2 * np.cos((2*k + 1) * np.pi / (2*(n + 1)))
xs = np.sort(xs)
ys = f_true(xs)

# Piecewise linear interpolant
def piecewise_linear(x):
    return np.interp(x, xs, ys)

P1 = piecewise_linear(xs)

# Global Lagrange interpolant
P_L = lagrange(xs, ys)
PL = P_L(xs)

# Local bubble function supported on each interval
def bubble(x, i):
    xL, xR = xs[i], xs[i + 1]
    h = xR - xL
    t = (x - xL) / h
    out = np.zeros_like(x)
    mask = (x >= xL) & (x <= xR)
    out[mask] = t[mask] * (1 - t[mask])
    return out

xx = np.linspace(a, b, 3000)
P1_xx = piecewise_linear(xx)
PL_xx = P_L(xx)

# Regularization parameter controlling alignment with Lagrange interpolation
lam = 1e-2
alphas = np.zeros(n)

# Optimize bubble coefficients by minimizing curvature and alignment error
for i in range(n):
    phi = bubble(xx, i)

    def objective(alpha):
        f = P1_xx + alpha * phi
        d2 = np.gradient(np.gradient(f, xx), xx)
        curvature = simpson(d2**2, xx)
        align = simpson((f - PL_xx)**2, xx)
        return curvature + lam * align

    grid = np.linspace(-5, 5, 400)
    vals = np.array([objective(a_) for a_ in grid])
    alphas[i] = grid[np.argmin(vals)]

# Curvature-corrected interpolant
def curvature_lagrange_interp(x):
    y = piecewise_linear(x)
    for i in range(n):
        y += alphas[i] * bubble(x, i)
    return y

xx = np.linspace(a, b, 2000)
y_true = f_true(xx)
y_lin  = piecewise_linear(xx)
y_lag  = P_L(xx)
y_curv = curvature_lagrange_interp(xx)

plt.figure(figsize=(10,6))
plt.plot(xx, y_true, 'k', lw=2, label="True")
plt.plot(xx, y_lag, ':', lw=2, label="Lagrange")
plt.plot(xx, y_curv, '-.', lw=2, label="Curvature-corrected")
plt.scatter(xs, ys, c='red', zorder=5, label="Nodes")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.bar(range(n), alphas)
plt.grid()
plt.xlabel("Interval index i")
plt.ylabel("alpha_i")
plt.title("Optimized bubble coefficients")
plt.show()

# Squared error comparison
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

