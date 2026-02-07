from manim import *
import numpy as np
from scipy.interpolate import lagrange
from scipy.integrate import simpson

# Experiment 4: Curvature-Corrected Lagrange Interpolation Visualization
#
# This script performs and visualizes the following steps:
# 1. Define the original function f(x) = 1 / (1 + x^2) over [a, b].
# 2. Generate Chebyshev interpolation nodes and compute corresponding function values.
# 3. Construct:
#    - Piecewise linear interpolation
#    - Lagrange interpolation
#    - Bubble basis functions for curvature correction
# 4. Compute curvature correction coefficients (alphas) for each interval by minimizing
#    the sum of squared second derivative (curvature) and deviation from Lagrange interpolation.
# 5. Define a corrected interpolation function:
#    P_curv(x) = piecewise_linear(x) + sum(alpha_i * phi_i(x))
# 6. Prepare scalar versions of functions for Manim plotting.
# 7. Construct a Manim scene:
#    - Display explanation text and axes
#    - Plot original function, interpolation nodes, piecewise linear and Lagrange interpolations
#    - Overlay curvature-corrected interpolation
#    - Compute and optionally show error metrics
#    - Present step-by-step visualization of alpha_i growth for selected intervals
#      using ValueTracker and always_redraw, synchronized with explanatory text
# 8. Show both the left-side explanation and right-side sample axes with local interval plots
#    to illustrate how curvature correction improves interpolation locally.

# Original function
def f_true(x):
    return 1 / (1 + x**2)

# Parameters
a, b = -5.0, 5.0
n = 10

# Chebyshev interpolation nodes
k = np.arange(n + 1)
xs = (a+b)/2 + (b-a)/2 * np.cos((2*k+1)*np.pi/(2*(n+1)))
xs = np.sort(xs)
ys = f_true(xs)

# Piecewise linear interpolation using the Chebyshev nodes
def piecewise_linear(x):
    return np.interp(x, xs, ys)

# Lagrange interpolation
P_L = lagrange(xs, ys)

# Bubble basis function for each interval
def bubble(x, i):
    xL, xR = xs[i], xs[i+1]
    h = xR - xL
    t = (x - xL)/h
    out = np.zeros_like(x)
    mask = (x >= xL) & (x <= xR)
    out[mask] = t[mask]*(1 - t[mask])
    return out

# For each interval, compute alpha_i to minimize curvature and deviation from Lagrange interpolation
xx = np.linspace(a, b, 3000)
P1_xx = piecewise_linear(xx)
PL_xx = P_L(xx)
lam = 1e-2
alphas = np.zeros(n)

for i in range(n):
    phi = bubble(xx, i)
    def objective(alpha):
        f = P1_xx + alpha*phi
        d2 = np.gradient(np.gradient(f, xx), xx)
        curvature = simpson(d2**2, xx)
        align = simpson((f - PL_xx)**2, xx)
        return curvature + lam*align
    grid = np.linspace(-5, 5, 400)
    vals = np.array([objective(a_) for a_ in grid])
    alphas[i] = grid[np.argmin(vals)]

def curvature_lagrange_interp(x):
    y = piecewise_linear(x)
    for i in range(n):
        y += alphas[i]*bubble(x, i)
    return y

# Convert vectorized functions to scalar functions for Manim's plot interface
def P_L_scalar(x):
    return float(P_L(x))

def curvature_scalar(x):
    return float(curvature_lagrange_interp(np.array([x])))

xx_plot = np.linspace(a, b, 2000)
y_true = f_true(xx_plot)
y_lin  = piecewise_linear(xx_plot)
y_lag  = P_L(xx_plot)
y_curv = curvature_lagrange_interp(xx_plot)

# Manim Scene
class CurvatureCorrectionScene(Scene):
    def construct(self):
        # Set background color to white
        self.camera.background_color = WHITE

        # Explanation text at top
        explanation = Text(
            "Experiment 4:\nCurvature-Corrected Lagrange Interpolation",
            font_size=36,
            color=BLACK
        )
        explanation.to_edge(ORIGIN)

        # Animate explanation
        self.play(FadeIn(explanation), run_time=1)
        self.wait(2)
        self.play(FadeOut(explanation), run_time=1)

        # Create axes
        axes = Axes(
            x_range=[a, b, 1],
            y_range=[0, 1.2, 0.2],
            x_length=10,
            y_length=5,
            axis_config={
                "include_tip": False,
                "color": BLACK,
                "stroke_width": 2,
            },
        )
        self.add(axes)
        self.play(Create(axes), run_time=0.6)

        # Plot original function
        func_plot = axes.plot(lambda x: 1/(1+x**2), color=BLUE)
        func_label = MathTex(r"f(x) = \frac{1}{1+x^2}", color=BLACK).move_to([0,3.3,0])
        self.play(Create(func_plot), Write(func_label), run_time=0.8)

        # Display interpolation nodes
        node_dots = [Dot(axes.coords_to_point(x, y), color=RED) for x, y in zip(xs, ys)]
        self.play(*[Create(dot) for dot in node_dots], run_time=0.5)
        self.wait(0.5)

        # Plot piecewise linear interpolation
        linear_plot = VGroup()
        for i in range(n):
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = ys[i], ys[i+1]
            segment = Line(
                axes.coords_to_point(x0, y0),
                axes.coords_to_point(x1, y1),
                color=GREEN
            )
            linear_plot.add(segment)
        linear_label = MathTex(r"L_\text{linear}(x)", color=BLACK).next_to(linear_plot.get_top(), UP)
        self.play(Create(linear_plot), Write(linear_label), run_time=1)
        current_label = linear_label

        # Plot Lagrange interpolation
        lag_plot = axes.plot(P_L_scalar, color=ORANGE)
        lag_label = MathTex(r"P_\text{Lagrange}(x)", color=BLACK).next_to(lag_plot.get_top(), UP)
        self.play(FadeOut(current_label), run_time=0.8)
        self.play(Create(lag_plot), Write(lag_label), run_time=1.2)
        current_label = lag_label

        # Plot curvature-corrected interpolation
        curv_plot = axes.plot(curvature_scalar, color=PURPLE)
        curv_label = MathTex(r"P_\text{curv}(x)", color=BLACK).next_to(curv_plot.get_top(), UP)
        self.play(FadeOut(current_label), run_time=0.8)
        self.play(FadeOut(linear_plot), Create(curv_plot), Write(curv_label), run_time=1.2)
        self.wait(3)
        current_label = curv_label

        # Compute errors and improvement
        xx_plot = np.linspace(a, b, 2000)
        y_true = f_true(xx_plot)
        y_lin  = piecewise_linear(xx_plot)
        y_lag  = P_L(xx_plot)
        y_curv = curvature_lagrange_interp(xx_plot)

        err_lin  = simpson((y_true - y_lin)**2, xx_plot)
        err_lag  = simpson((y_true - y_lag)**2, xx_plot)
        err_curv = simpson((y_true - y_curv)**2, xx_plot)
        improvement = (err_lin - err_curv) / err_lin * 100

        self.play(
            FadeOut(func_plot),
            FadeOut(lag_plot),
            FadeOut(curv_plot),
            FadeOut(curv_label),
            FadeOut(func_label),
            FadeOut(axes),
            *[FadeOut(dot) for dot in node_dots],
            run_time=0.8
        )
        text_group = VGroup()

        # First line: piecewise linear interpolation
        text1 = MathTex(r"L_\text{linear}(x)", font_size=36)
        text_group.add(text1).move_to([-5,1.4,0])

        # Second line: curvature-corrected interpolation
        text2 = MathTex(r"P_\text{curv}(x) = L_\text{linear}(x) + \alpha \phi(x)", font_size=36)
        text2.next_to(text1, DOWN, buff=0.5)
        text_group.add(text2)

        # Third line: definition of correction coefficient α_i
        text3 = MathTex(
            r"\alpha_i = \arg\min_\alpha \left[ \int (f''(x))^2 dx + \lambda \int (P_\text{curv}(x) - P_\text{Lagrange})^2 dx \right]",
            font_size=28
        )
        text3.next_to(text2, DOWN, buff=0.5)
        text_group.add(text3)

        # Bubble basis function definition
        text4 = MathTex(r"\phi_i(x) = t(1-t),\ t = \frac{x - x_i}{x_{i+1}-x_i}", font_size=36)
        text4.next_to(text3, DOWN, buff=0.5)
        text_group.add(text4)

        # Arrange text on the left side
        text_group.arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT).shift(UP*1.5)

        # Add text to the scene
        self.add(text_group)
        self.play(*[Write(txt) for txt in text_group])
        self.wait(0.5)

        # Sample axes on the right side showing the function and Lagrange interpolation
        interval_indices = [5, 6, 7]  # Indices of intervals to display
        x_min, x_max = xs[interval_indices[0]], xs[interval_indices[-1]+1]
        y_min, y_max = min(ys[interval_indices[0]:interval_indices[-1]+2])-0.05, max(ys[interval_indices[0]:interval_indices[-1]+2])+0.05

        axes_sample = Axes(
            x_range=[x_min, x_max, (x_max-x_min)/5],
            y_range=[y_min, y_max, (y_max-y_min)/5],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False, "color": BLACK},
        ).to_edge(RIGHT)
        self.add(axes_sample)
        self.play(Create(axes_sample))

        # Plot the original function
        def f_true_scalar(x):
            return float(f_true(np.array([x])))
        true_plot = axes_sample.plot(f_true_scalar, color=BLUE)
        self.play(Create(true_plot))
        self.wait(0.5)

        # Plot piecewise linear interpolation
        def piecewise_linear_scalar_multi(x):
            return float(piecewise_linear(np.array([x])))
        line_plot = axes_sample.plot(piecewise_linear_scalar_multi, color=GREEN)
        self.play(Create(line_plot))
        self.wait(0.5)

        # Plot interval Lagrange interpolation
        def lagrange_scalar(x):
            return float(P_L(x))
        lag_plot = axes_sample.plot(lagrange_scalar, color=ORANGE)
        self.play(Create(lag_plot))
        self.wait(0.5)

        # Animate curvature correction: increase alpha_i for each interval while updating the plot
        # Local bubble basis function for each interval
        def bubble_scalar_multi(x, interval_idx):
            x0, x1 = xs[interval_idx], xs[interval_idx + 1]
            h = x1 - x0
            if x < x0 or x > x1:
                return 0.0
            t = (x - x0) / h
            return t * (1 - t)

        # Alpha trackers for each interval
        alpha_trackers = [ValueTracker(0) for _ in interval_indices]

        # Single function including correction
        # Initially matches the piecewise linear interpolation exactly
        corrected_curve = always_redraw(
            lambda: axes_sample.plot(
                lambda x: piecewise_linear_scalar_multi(x)
                          + sum(
                              alpha_trackers[k].get_value() * bubble_scalar_multi(x, interval_indices[k])
                              for k in range(len(interval_indices))
                          ),
                color=PURPLE,
                stroke_width=6
            )
        )

        # Add corrected curve to scene
        self.add(corrected_curve)

        # Sequentially animate each alpha_i coefficient on the same curve for visualization
        for k, i in enumerate(interval_indices):
            # Highlight the corresponding text line (for explanation)
            highlight_colors = [BLACK] * len(text_group)
            if k < len(text_group):
                highlight_colors[k] = RED

            self.play(
                *[txt.animate.set_color(c) for txt, c in zip(text_group, highlight_colors)],
                run_time=0.2
            )

            # Animate the growth of the correction coefficient for this interval
            self.play(
                alpha_trackers[k].animate.set_value(alphas[i]),
                run_time=2,
                rate_func=linear
            )

        # Restore text color to black
        self.play(
            *[txt.animate.set_color(BLACK) for txt in text_group],
            run_time=0.3
        )
        self.wait(2)
