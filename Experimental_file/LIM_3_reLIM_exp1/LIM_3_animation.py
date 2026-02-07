from manim import *
import numpy as np
from scipy.integrate import quad

class Experiment_1(MovingCameraScene):

# ============================================================
# Experiment 1: Subdivided Re-interpolated Lagrange Method
#
# This script presents a numerical experiment illustrating a
# subdivided re-interpolated Lagrange method for the function
#     f(x) = 1 / (1 + x^2)
# on the interval [-5, 5].
#
# Instead of constructing a single high-degree polynomial over
# the entire interval, an initial interpolation is first
# performed. Subintervals exhibiting large interpolation errors
# are then identified and locally re-interpolated using
# additional interpolation nodes.
#
# The interpolation error is evaluated as
#     E = ∫ |f(x) − P_n(x)| dx,
# and the error contribution of each subinterval is visualized.
# Local refinement is applied when the error exceeds a prescribed
# threshold.
#
# Camera zooming and translation are used to highlight the
# localized re-interpolation process and the resulting
# improvement in approximation accuracy.
# ============================================================

    def construct(self):

        self.camera.background_color = WHITE

        Introduction = Text(
            "Experiment 1: \n分割再補間ラグランジュ法"
            , color=BLACK
            , font= "Hiragino Mincho ProN"
            , line_spacing=1.5
            ).scale(0.8)
        
        Explain = Text("分割再補間ラグランジュ法:\n全体を一気に補間せず、粗く補間した後、\n誤差の大きい部分を局所的に再補間する方法"
                       , color=BLACK
                       , font="Hiragino Mincho ProN"
                       , line_spacing=1
                    ).scale(1)
        
        Error_text = Text("もしE≥10^-6ならば、補間点を3つ増やし,\n端点含めた5点で再補間を行う。"
                          , color=BLACK 
                          , font= "Hiragino Mincho ProN"
                          ,line_spacing=1.2).scale(0.8)

        # --- Define function and Chebyshev nodes ---
        def f(x):
            return 1 / (1 + x**2)

        def chebyshev_nodes(k, n):
            return np.cos((2 * k + 1) * np.pi / (2 * n))

        func = MathTex(
            r"f(x) = \frac{1}{1+x^2}\quad[-5,5]", 
            color=BLACK)

        Error = MathTex(
            r"E = \int_a^b \left| f(x) - P_n(x) \right| \, dx",
            color=BLACK
        )

        func.move_to([0, 2, 0])
        Error.move_to([0, 2, 0])
        Error_text.move_to([0, 1.8, 0])

        for mob in [func, Error, Introduction, Error_text, Explain]:
            mob.set_color(BLACK)
            mob.set_stroke(BLACK, width=1)

        def Lag_int(x, x_nodes, y_nodes):
            n = len(x_nodes)
            total = 0
            for i in range(n):
                xi, yi = x_nodes[i], y_nodes[i]
                term = yi
                for j in range(n):
                    if j != i:
                        xj = x_nodes[j]
                        term *= (x - xj) / (xi - xj)
                total += term
            return total

        # --- Create nodes ---
        n = 11
        x_nodes = np.array([5 * chebyshev_nodes(k, n) for k in range(n)])
        y_nodes = f(x_nodes)
        max_refine_points = 5

        def Lag_int_1(x):
            return Lag_int(x, x_nodes, y_nodes)

        # --- Setup axes ---
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-0.5, 1.5, 0.5],
            x_length=12,
            y_length=4.5,
            axis_config={"color": BLACK},
            tips=False,
        ).scale(0.9)
        axes.move_to([0, -1.5, 0])

        x_label, y_label = axes.get_axis_labels("x", "y")
        x_label.set_color(BLACK)
        y_label.set_color(BLACK)
        axes.add(x_label, y_label)
        axes.add_coordinates()
        for label in axes.x_axis.numbers + axes.y_axis.numbers:
            label.set_color(BLACK)

        def initial_points(x_nodes):
            return VGroup(*[Dot(axes.c2p(xi, f(xi)), color=ORANGE, radius=0.08) for xi in x_nodes])
        initial_nodes = initial_points(x_nodes)  

        # --- Plot function and interpolation ---
        graph = axes.plot(f, color=BLUE, stroke_width=5)
        Lag_graph_1 = axes.plot(Lag_int_1, color=RED, stroke_width=5)

        # --- Error area polygons (f and interpolation) ---
        error_areas = VGroup()
        for i in range(len(x_nodes) - 1):
            a, b = x_nodes[i], x_nodes[i + 1]
            x_vals = np.linspace(a, b, 100)
            points_upper = [axes.c2p(x, f(x)) for x in x_vals]
            points_lower = [axes.c2p(x, Lag_int_1(x)) for x in reversed(x_vals)]
            
            poly = Polygon(
                *points_upper, *points_lower,
                color=PURE_GREEN,
                fill_opacity=0.7,
                stroke_width=0
            )
            error_areas.add(poly)

        # --- Animate the base interpolation and error ---
        self.play(FadeIn(Introduction), run_time=0.5)
        self.wait(2.6)
        self.play(FadeOut(Introduction), run_time=0.8)
        self.play(FadeIn(Explain),run_time=1)
        self.wait(5)
        self.play(FadeOut(Explain),run_time=0.5)
        self.play(FadeIn(axes, func, initial_nodes))
        self.play(Create(graph), run_time=2.5)
        self.play(Create(Lag_graph_1), FadeOut(func),run_time=2.5)
        self.wait(1)
       　
       # --- Animate error areas sequentially with integral formula ---
        for i, poly in enumerate(error_areas):
            a, b = x_nodes[i], x_nodes[i+1]

            # --- Create new integral formula for current interval ---
            new_error = MathTex(
                r"E = \int_{{{:.2f}}}^{{{:.2f}}} \left( f(x) - P_n(x) \right)^2 \, dx".format(a, b),
                color=BLACK
            ).move_to(Error.get_center()).set_color(BLACK).set_stroke(BLACK, width=1)

            # --- Remove old formula if exists ---
            if i != 0:
                self.play(FadeOut(Error), run_time=0.2)

            # --- Add new formula ---
            self.play(Write(new_error),Write(poly),run_time=0.5)
            Error = new_error
        
        self.play(FadeOut(Error), run_time=0.2)
        self.play(FadeIn(Error_text), run_time=2)
        self.wait(2.5)
        self.play(FadeOut(Error_text), run_time=0.5)

        # --- Refined interpolation with camera zoom once, then parallel moves ---
        zoom_scale = 0.4
        zoom_done = False

        for i in range(len(x_nodes) - 1):
            a, b = x_nodes[i], x_nodes[i + 1]
            x_center = (a + b) / 2
            y_center = f(x_center)

            # --- Generate refined interpolation for this interval ---
            x_sub = np.linspace(a, b, max_refine_points)
            y_sub = f(x_sub)

            def Lag_refined(x, x_sub=x_sub, y_sub=y_sub):
                return Lag_int(x, x_sub, y_sub)

            x_vals = np.linspace(a, b, 50)
            points = [axes.c2p(x, Lag_refined(x)) for x in x_vals]
            Lag_graph_sub = VMobject(color=PURPLE, stroke_width=6)
            Lag_graph_sub.set_points_smoothly(points)

            dots = VGroup(*[Dot(axes.c2p(xi, f(xi)), color=ORANGE, radius=0.04) for xi in x_sub])

            # --- Camera animation: zoom once, then move only ---
            if not zoom_done:
                camera_animation = self.camera.frame.animate.move_to(axes.c2p(x_center, y_center)).scale(zoom_scale)
                zoom_done = True
            else:
                camera_animation = self.camera.frame.animate.move_to(axes.c2p(x_center, y_center))

            self.play(camera_animation, run_time=0.55)
            self.play(Create(dots), run_time=0.3)
            self.play(Create(Lag_graph_sub), run_time=0.4)

        # --- Final camera zoom out to exact original axis center ---
        self.wait(0.8)
        self.play(
            self.camera.frame.animate.scale(1 / zoom_scale).move_to(axes.c2p(0, 1.25)),
            run_time=1
        )
        self.wait(3)
