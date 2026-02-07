import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Curvature-controlled interpolation via iterative slope refinement.
# Initial slopes are estimated from neighboring data points and
# iteratively updated to match the derivative of the interpolant.
# This self-consistent process reduces curvature artifacts
# while preserving interpolation accuracy at the nodes.

def f_true(x):
    return 1 / (1 + x**2)

a, b = -5.0, 5.0
n = 10

# Chebyshev nodes on [a, b]
k = np.arange(n + 1)
xs = (a + b)/2 + (b - a)/2 * np.cos((2*k + 1)*np.pi/(2*(n+1)))
xs = np.sort(xs)
ys = f_true(xs)

def piecewise_linear(x, xs, ys):
    return np.interp(x, xs, ys)

def lagrange_eval(x, xs, ys):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    for i in range(len(xs)):
        Li = np.ones_like(x)
        for j in range(len(xs)):
            if i != j:
                Li *= (x - xs[j]) / (xs[i] - xs[j])
        y += ys[i] * Li
    return y

# Finite-difference slope initialization
def initial_slopes(xs, ys):
    n = len(xs)
    m = np.zeros(n)
    for i in range(n):
        if i == 0:
            m[i] = (ys[1] - ys[0]) / (xs[1] - xs[0])
        elif i == n - 1:
            m[i] = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        else:
            m[i] = (ys[i+1] - ys[i-1]) / (xs[i+1] - xs[i-1])
    return m

# Piecewise cubic interpolant defined by nodal values and slopes
class CubicSlopeSpline:
    def __init__(self, xs, ys, ms):
        self.xs = xs
        self.n = len(xs) - 1
        self.coeffs = []

        for i in range(self.n):
            h = xs[i+1] - xs[i]
            y0, y1 = ys[i], ys[i+1]
            m0, m1 = ms[i], ms[i+1]

            a = (2*(y0 - y1) + h*(m0 + m1)) / h**3
            b = (3*(y1 - y0) - h*(2*m0 + m1)) / h**2
            c = m0
            d = y0

            self.coeffs.append((a, b, c, d))

    def __call__(self, x):
        x = np.asarray(x)
        y = np.zeros_like(x)
        for i in range(self.n):
            mask = (self.xs[i] <= x) & (x <= self.xs[i+1])
            dx = x[mask] - self.xs[i]
            a, b, c, d = self.coeffs[i]
            y[mask] = ((a*dx + b)*dx + c)*dx + d
        return y

    def derivative(self):
        return CubicSlopeSplineDerivative(self)
    

class CubicSlopeSplineDerivative:
    def __init__(self, spline):
        self.xs = spline.xs
        self.coeffs = spline.coeffs
        self.n = spline.n

    def __call__(self, x):
        x = np.asarray(x)
        y = np.zeros_like(x)
        for i in range(self.n):
            mask = (self.xs[i] <= x) & (x <= self.xs[i+1])
            dx = x[mask] - self.xs[i]
            a, b, c, _ = self.coeffs[i]
            y[mask] = (3*a*dx + 2*b)*dx + c
        return y

# Iterative slope self-consistency loop
def iterated_slope_spline(xs, ys, max_iter=20, tol=1e-10):
    m = initial_slopes(xs, ys)

    for _ in range(max_iter):
        spline = CubicSlopeSpline(xs, ys, m)
        m_new = spline.derivative()(xs)

        # Fix endpoint slopes
        m_new[0]  = m[0]
        m_new[-1] = m[-1]

        if np.linalg.norm(m_new - m) < tol:
            break
        m = m_new

    return spline

S_iter = iterated_slope_spline(xs, ys)

xx = np.linspace(a, b, 2000)

y_true  = f_true(xx)
y_lag   = lagrange_eval(xx, xs, ys)
y_slope = S_iter(xx)

def L_inf(y, y_ref):
    return np.max(np.abs(y - y_ref))

def L2(y, y_ref, x):
    return np.sqrt(simpson((y - y_ref)**2, x))

def area_squared_error(y, y_ref, x):
    return simpson((y - y_ref)**2, x)

print("=== Error comparison ===")
print(f"Lagrange            L∞ = {L_inf(y_lag, y_true):.3e},  L2 = {L2(y_lag, y_true, xx):.3e},  Area = {area_squared_error(y_lag, y_true, xx):.3e}")
print(f"Iterated slope      L∞ = {L_inf(y_slope, y_true):.3e},  L2 = {L2(y_slope, y_true, xx):.3e},  Area = {area_squared_error(y_slope, y_true, xx):.3e}")

plt.figure(figsize=(10,6))
plt.plot(xx, y_true, 'k', linewidth=2, label="True function")
plt.plot(xx, y_lag, ':', label="Lagrange")
plt.plot(xx, y_slope, '-.', label="Iterated slope spline")
plt.scatter(xs, ys, c='red', zorder=5, label="Nodes")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.semilogy(xx, np.abs(y_lag - y_true), label="Lagrange error")
plt.semilogy(xx, np.abs(y_slope - y_true), label="Slope spline error")
plt.grid()
plt.legend()
plt.show()
