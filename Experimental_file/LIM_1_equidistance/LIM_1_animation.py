from manim import *
import numpy as np

class Runge_phenomenon(Scene):
# ============================================================
# Visualization of the Runge Phenomenon Using Polynomial Interpolation
#
# This script presents a visualization of the Runge phenomenon
# occurring in polynomial interpolation on the interval [-5, 5].
# The target function is defined as
#     f(x) = 1 / (1 + x^2).
#
# The function is interpolated at equally spaced nodes by means
# of a Lagrange interpolation polynomial. Although the interpolant
# matches the function values exactly at the interpolation nodes,
# it exhibits pronounced oscillatory behavior near the endpoints
# of the interval as the polynomial degree increases.
#
# The visualization includes:
#   - The original function f(x), shown in blue.
#   - The interpolation nodes, shown as green markers.
#   - The resulting Lagrange interpolation polynomial, shown in red.
#
# Elliptical markers and directional arrows are employed to
# highlight the endpoint oscillations, thereby illustrating
# a fundamental limitation of high-degree polynomial interpolation
# when uniformly spaced nodes are used.
# ============================================================
    def construct(self):
        self.camera.background_color = WHITE

        def f(x):
            return 1 / (1 + x**2)

        func = MathTex(r"f(x) = \frac{1}{1 + x^2}\quad[-5,5]").scale(0.9)
        T1 = Text("ルンゲ現象", font="Hiragino Mincho ProN", color=BLACK).scale(1)

        for mob in [func, T1]:
            mob.set_color(BLACK)
            mob.set_stroke(BLACK, width=1)

        # --- axes ---
        axes = Axes(
            x_range=[-6, 6, 1],
            y_range=[-0.5, 1.5, 0.5],
            x_length=12,
            y_length=4.5,
            axis_config={"color": BLACK},
            tips=False,
        ).scale(0.9)

        axes.move_to([0, -1, 0])

        # -– axis labels ---
        x_labels, y_labels = axes.get_axis_labels(x_label="x", y_label="y")
        x_labels.set_color(BLACK)
        y_labels.set_color(BLACK)
        axes.add(x_labels, y_labels)
        axes.add_coordinates()
        for label in axes.x_axis.numbers + axes.y_axis.numbers:
            label.set_color(BLACK)

        # --function graph---
        graph = axes.plot(f, color=BLUE, stroke_width=5)

        # --- Interpolation points ---
        x_values = np.linspace(-5, 5, 11)
        y_values = f(x_values)
        dots = VGroup(*[
            Dot(axes.c2p(x, f(x)), color=GREEN, radius=0.12)
            for x in x_values
        ])

        # --- lagrange interpolation ---
        def Lag_int(x):
            total = 0
            n = len(x_values)
            for i in range(n):
                xi, yi = x_values[i], y_values[i]
                term = yi
                for j in range(n):
                    if j != i:
                        xj = x_values[j]
                        term *= (x - xj) / (xi - xj)
                total += term
            return total

        Lag_graph = axes.plot(Lag_int, color=RED_D, stroke_width=5)

        # ---emphasis shapes---
        left_ellipse = Ellipse(width=1.7, height=6.1, color=BLACK).move_to([-4.2, -0.9, 0])
        right_ellipse = Ellipse(width=1.7, height=6.1, color=BLACK).move_to([4.2, -0.9, 0])
        arrow_right = Arrow(start=[1.4, -3.5, 0], end=[3.8, -3.5, 0], color=BLACK, stroke_width=2)
        arrow_left = Arrow(start=[-1.4, -3.5, 0], end=[-3.8, -3.5, 0], color=BLACK, stroke_width=2)

        # --- text position ---
        func.move_to([0, 2, 0])
        T1.move_to([0, -3.5, 0])

        # ---animation---
        self.play(Write(func), Create(axes), Create(graph), run_time=3)
        self.wait(3.4)
        self.play(Create(dots), run_time=3)
        self.wait(2)
        self.play(Create(Lag_graph), run_time=3)
        self.play(Create(left_ellipse), Create(right_ellipse), run_time=1)
        self.play(Create(arrow_right), Create(arrow_left), Write(T1), run_time=1.5)
        self.wait(5)
