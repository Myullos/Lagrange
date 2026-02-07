from manim import *
import numpy as np

class Experiment_2(MovingCameraScene):

# ============================================================
# Experiment 2: Lagrange Interpolation with Optimized y-Values
#
# This experiment studies a modified Lagrange interpolation
# scheme in which the x-coordinates of the interpolation nodes
# are fixed at Chebyshev nodes, while only the y-values are
# treated as optimization variables.
#
# Starting from the exact values of
#     f(x) = 1 / (1 + x^2)
# at 11 Chebyshev nodes on the interval [-5, 5], random local
# perturbations are applied to the y-values. For each candidate,
# a new interpolation polynomial is constructed.
#
# The approximation error is evaluated using the squared L2 norm
#     E = ∫_{-5}^{5} (f(x) − P_n(x))^2 dx,
# numerically approximated by the trapezoidal rule.
#
# A stochastic local search is performed around the current best
# solution, retaining the perturbation that yields the smallest
# error. The animation visualizes how local adjustments of node
# values affect the interpolation polynomial and reduce the error.
# ============================================================

    def construct(self):

        # === SCENE SETUP ===
        self.camera.background_color = WHITE

        # --- Introduction text ---
        Introduction = Text(
            "Experiment 2: \n y座標可変式ラグランジュ補間法",
            color=BLACK,
            font="Hiragino Mincho ProN",
            line_spacing=1.5
        ).scale(0.8)

        Inp = Text(
            "11点の補間点のy座標を±10^-3分だけ変化させ、\n誤差Eが最小となる補間多項式を探索する。",
            color=BLACK,
            font="Hiragino Mincho ProN",
            line_spacing=1).scale(0.8)

        Conc = Text("これを1000回繰り返し、\n最も誤差Eが小さくなる補間多項式を求める。",
            color=BLACK,
            font="Hiragino Mincho ProN",
            line_spacing=1).scale(0.8)
        
        # --- Define target function ---
        def f(x):
            return 1 / (1 + x**2)

        # --- Define Chebyshev nodes ---
        def chebyshev_nodes(k, n):
            return np.cos((2 * k + 1) * np.pi / (2 * n))

        # --- Function formula ---
        func = MathTex(r"f(x) = \frac{1}{1 + x^2}\quad[-5,5]").set_color(BLACK)
        
        # --- Numerical approximation of the squared L2 error using the trapezoidal rule ---
        error_formula = MathTex(r"E = \int_{-5}^{5} \left( f(x) - P_n(x) \right)^2 \, dx"
            , color=BLACK
        )

        # --- Lagrange interpolation polynomial ---
        def Lag_int(x, x_nodes, y_nodes):
            n = len(x_nodes)
            L = 0
            for i in range(n):
                term = y_nodes[i]
                for j in range(n):
                    if j != i:
                        term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
                L += term
            return L

        # === DATA PREPARATION ===
        n = 11
        x_nodes = np.array([5 * chebyshev_nodes(k, n) for k in range(n)])
        y_nodes = f(x_nodes)

        def Lag_int_1(x):
            return Lag_int(x, x_nodes, y_nodes)

        # === AXES SETUP ===
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-0.5, 1.5, 0.5],
            x_length=12,
            y_length=4.5,
            axis_config={"color": BLACK},
            tips=False,
        ).scale(0.9)
        axes.move_to([0, -1.5, 0])

        # --- Axis labels ---
        x_label, y_label = axes.get_axis_labels("x", "y")
        x_label.set_color(BLACK)
        y_label.set_color(BLACK)
        axes.add(x_label, y_label)
        axes.add_coordinates()
        for label in axes.x_axis.numbers + axes.y_axis.numbers:
            label.set_color(BLACK)

        # --- Create initial interpolation points ---
        def create_points(x_nodes, y_nodes):
            return VGroup(*[Dot(axes.c2p(xi, yi), color=ORANGE, radius=0.08) for xi, yi in zip(x_nodes, y_nodes)])

        points = create_points(x_nodes, y_nodes)

        func.move_to([3, 1.8, 0])
        error_formula.move_to([-3, 1.8, 0])
        Inp.move_to([0, 1.6, 0])
        Conc.move_to([0, 1.6, 0])

        # --- Create function and interpolation graphs ---
        graph = axes.plot(f, color=BLUE, stroke_width=5)
        Lag_graph_1 = axes.plot(Lag_int_1, color=RED, stroke_width=5)

        for mob in [func, Introduction, error_formula, Inp, Conc]:
            mob.set_color(BLACK)
            mob.set_stroke(BLACK, width=1)

        # === INITIAL DISPLAY ===
        self.play(FadeIn(Introduction))
        self.wait(2.2)
        self.play(FadeOut(Introduction))
        self.add(axes, func, points, error_formula)
        self.play(Create(graph), run_time=2.5)
        self.play(Create(Lag_graph_1), FadeOut(func,error_formula),run_time=1)
        self.play(FadeIn(Inp),run_time=1.5)
        self.wait(4)
        self.play(FadeOut(Inp))
        self.play(FadeIn(error_formula, func))
        original_frame = self.camera.frame.copy()

        rightmost_index = np.argmax(x_nodes)
        target_coord = axes.c2p(x_nodes[rightmost_index], y_nodes[rightmost_index])
        rightmost_dot = Dot(target_coord, color=ORANGE, radius=0.1)
        self.add(rightmost_dot)

        # --- Zoom into that point ---
        self.play(
            self.camera.frame.animate.scale(0.3).move_to(target_coord),
            run_time=2
        )
        self.wait(0.5)

        # --- Define real animation amplitude (not ±10^-3) ---
        visual_delta = 1e-1
        #label_delta = 1e-3       
        up_y = y_nodes[rightmost_index] + visual_delta
        down_y = y_nodes[rightmost_index] - visual_delta
        up_pos = axes.c2p(x_nodes[rightmost_index], up_y)
        down_pos = axes.c2p(x_nodes[rightmost_index], down_y)

        # --- Draw double arrow representing visible motion range ---
        arrow_group = DoubleArrow(
            start=down_pos,
            end=up_pos,
            color=PURE_BLUE,
            stroke_width=4,
            tip_length=0.4
        )

        # --- Label shows theoretical ±10⁻³ ---
        label_text = MathTex(r"\pm 10^{-3}", color=BLACK).scale(0.8)
        label_text.next_to(arrow_group, RIGHT, buff=0.1)

        # --- Show arrow + label ---
        self.play(FadeIn(arrow_group), FadeIn(label_text), run_time=1)
        self.wait(0.3)

        # --- Move the dot up and down with large visible amplitude ---
        for pos in [up_pos, down_pos, target_coord]:
            self.play(rightmost_dot.animate.move_to(pos), run_time=0.8)
        self.wait(0.3)

        # --- Clean up + zoom out ---
        self.play(
            FadeOut(arrow_group),
            FadeOut(label_text),
            self.camera.frame.animate.become(original_frame),
            run_time=2
        )
        self.wait(0.5)

        # === PARAMETERS ===
        delta = 0.02      # # Random variation in y-values (exaggerated for visualization)
        num_steps = 10     # Number of random trials ( real experiment is performed 1000 times)
        Best_E = float("inf")  # Initialize best error
        Best_y_nodes = y_nodes.copy()

        # --- Display persistent Best_E label (always visible) ---
        # Use LaTeX \infty to properly render infinity symbol
        best_text = MathTex(r"Best\_E = \infty", color=PURE_BLUE).scale(1.5)
        best_text.move_to([3, 0, 0])
        self.add(best_text)

        # === MAIN LOOP ===
        # --- Stochastic local search around the current best interpolation ---
        # Note: Only the y-values of the interpolation nodes are optimized,
        # while the x-values (Chebyshev nodes) are kept fixed.
        for step in range(num_steps):
            # --- Always start from the best nodes so far ---
            y_nodes = Best_y_nodes.copy()

            # --- Apply random local perturbations around the current best solution ---
            new_y_nodes = [yi + np.random.uniform(-delta, delta) for yi in y_nodes]

            # --- Define new interpolation ---
            def Lag_int_new(x):
                return Lag_int(x, x_nodes, new_y_nodes)

            new_Lag_graph = axes.plot(Lag_int_new, color=RED, stroke_width=5)

            # --- Compute error area and total error ---
            x_vals = np.linspace(-5, 5, 300)
            upper = [axes.c2p(x, f(x)) for x in x_vals]
            lower = [axes.c2p(x, Lag_int_new(x)) for x in reversed(x_vals)]
            error_area = Polygon(*upper, *lower, color=GREEN, fill_opacity=0.4, stroke_width=0)
            
            # --- Compute squared L2 interpolation error over the full interval ---
            error_value = np.trapezoid((np.array([f(x) - Lag_int_new(x) for x in x_vals]))**2, x_vals)
            error_text = MathTex(
                f"E = {error_value:.5f}", color=BLACK
                ).scale(0.7).next_to(best_text, DOWN).scale(1.6)

            # --- Animate movement and update graph ---
            self.play(
                *[p.animate.move_to(axes.c2p(xi, yi)) for p, xi, yi in zip(points, x_nodes, new_y_nodes)],
                Transform(Lag_graph_1, new_Lag_graph),
                FadeIn(error_area),
                FadeIn(error_text),
                run_time=0.5
            )

            # --- Update best error (persistent) ---
            if error_value < Best_E:
                Best_E = error_value
                Best_y_nodes = np.array(new_y_nodes)
                # Update persistent label only
                new_best_text = MathTex(f"Best_E = {Best_E:.5f}", color=PURE_BLUE).scale(1.2).move_to(best_text)
                self.play(Transform(best_text, new_best_text), run_time=0.5)

            # --- Fade out temporary elements ---
            self.play(FadeOut(error_area), FadeOut(error_text), run_time=0.4)

        self.play(
            *[p.animate.move_to(axes.c2p(xi, yi)) for p, xi, yi in zip(points, x_nodes, Best_y_nodes)],
            run_time=1.5
        )

        # Draw final best interpolation curve
        final_graph = axes.plot(lambda x: Lag_int(x, x_nodes, Best_y_nodes), color=PURPLE, stroke_width=6)
        x_vals = np.linspace(-5, 5, 300)
        upper = [axes.c2p(x, f(x)) for x in x_vals]
        lower = [axes.c2p(x, Lag_int(x, x_nodes, Best_y_nodes)) for x in reversed(x_vals)]
        final_error_area = Polygon(*upper, *lower, color=PURE_GREEN, fill_opacity=0.7, stroke_width=0)
        self.play(Transform(Lag_graph_1, final_graph), run_time=0.4)
        self.play(FadeIn(final_error_area), run_time=0.5)
        self.play(FadeOut(func,error_formula))
        self.play(FadeIn(Conc),run_time=3)
        self.wait(7)
