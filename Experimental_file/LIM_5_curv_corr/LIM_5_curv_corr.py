import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from numpy.polynomial.chebyshev import Chebyshev
import matplotlib.pyplot as plt

def main():
    # Curvature-based correction of Chebyshev Lagrange interpolation.
    # A global interpolant P(x) is constructed from Chebyshev nodes,
    # then corrected by adding t * w(x) * P''(x), where the coefficient t
    # is determined by minimizing an integral squared error.

    def f(x):
        return 1.0 / (1.0 + x**2)

    # Weight function for curvature correction
    def w(x):
        return -x**4 + 25 * x**2

    # Chebyshev nodes on [-5, 5]
    N = 11
    k = np.arange(1, N + 1)
    nodes_x = 5 * np.cos((2 * k - 1) * np.pi / (2 * N))
    nodes_y = f(nodes_x)

    limit_low = np.min(nodes_x)
    limit_high = np.max(nodes_x)

    # Global Lagrange interpolant (via Chebyshev series for stability)
    P_poly = Chebyshev.fit(nodes_x, nodes_y, N - 1, domain=[-5, 5])
    P2_poly = P_poly.deriv(2)

    def P(x):
        return P_poly(x)

    def P2(x):
        return P2_poly(x)

    # Piecewise linear interpolant for reference
    sort_idx = np.argsort(nodes_x)
    nodes_x_sorted = nodes_x[sort_idx]
    nodes_y_sorted = nodes_y[sort_idx]

    L_interp = interp1d(nodes_x_sorted, nodes_y_sorted, kind='linear')
    def L(x):
        return L_interp(x)

    print("計算を開始します...")

    # Numerator and denominator of the optimal correction coefficient
    def numer_func(x):
        return w(x) * (P(x) - L(x)) * P2(x)

    def denom_func(x):
        return (w(x) * P2(x))**2

    val_num, _ = quad(numer_func, limit_low, limit_high, points=nodes_x)
    val_den, _ = quad(denom_func, limit_low, limit_high, points=nodes_x)

    t_value = - val_num / val_den

    print(f"算出された t の値: {t_value:.9f}")

    # Error evaluation with and without curvature correction
    def error_with_correction(x):
        return (f(x) - (P(x) + t_value * w(x) * P2(x)))**2

    def error_without_correction(x):
        return (f(x) - P(x))**2

    final_integral, _ = quad(error_with_correction, -5, 5, points=nodes_x)
    integral_pure, _ = quad(error_without_correction, -5, 5, points=nodes_x)

    print(f"補正なし(t=0) の積分誤差: {integral_pure:.5f}")
    print(f"補正あり(t使用) の積分誤差: {final_integral:.5f}")

    improvement = (integral_pure - final_integral) / integral_pure * 100
    print(f"誤差減少率: {improvement:.2f}%")

    xs_plot = np.linspace(-5, 5, 2000)

    plt.figure(figsize=(10, 6))
    plt.plot(xs_plot, f(xs_plot), color="black", linewidth=2.5,
             label="Original function f(x)")
    plt.plot(xs_plot, P(xs_plot), "--", color="blue", linewidth=2,
             label="Lagrange interpolation P(x)")
    plt.plot(xs_plot, P(xs_plot) + t_value * w(xs_plot) * P2(xs_plot),
             "-.", color="red", linewidth=2,
             label="Corrected interpolation")
    plt.scatter(nodes_x, nodes_y, color="blue", zorder=5,
                label="Chebyshev nodes")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

