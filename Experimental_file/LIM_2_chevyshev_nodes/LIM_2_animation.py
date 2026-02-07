from manim import *
import numpy as np

class Chebyshevnodes(Scene):
# ============================================================
# Visualization of Chebyshev Nodes in Polynomial Interpolation
#
# This script illustrates the use of Chebyshev nodes in polynomial
# interpolation for the function
#     f(x) = 1 / (1 + x^2)
# on the interval [-5, 5].
#
# Polynomial interpolation based on equally spaced nodes is first
# shown, emphasizing the oscillatory behavior near the interval
# endpoints. Chebyshev nodes,
#     x_k = cos((2k+1)π / (2n)),
# are then introduced through a geometric construction using a
# semicircle and projected onto the x-axis.
#
# A Lagrange interpolation polynomial constructed from the
# Chebyshev nodes demonstrates a clear reduction of endpoint
# oscillations, highlighting their advantage in high-degree
# polynomial interpolation.
# ============================================================
    def construct(self):
        self.camera.background_color = WHITE

        # --- Define the function and Chebyshev nodes ---
        def f(x):
            return 1 / (1 + x**2)

        def chebyshev_node(k, n):
            return np.cos((2 * k + 1) * np.pi / (2 * n))

        chev_tex = MathTex(
        r"\text{Chebyshev nodes:}\\ x_k = \cos\left(\frac{(2k+1)\pi}{2n}\right)\\k=(0,1,2,...,n-1)",
        ).scale(1.7)

        Explained = Text("⚫︎=π/10(rad)").set_color(BLACK).scale(0.7)
        
        # --- Create mathematical formulas ---
        func = MathTex("f(x) = \\frac{1}{1+x^2}")
        chebyshev_nodes_tex = MathTex("x_k = \\cos\\left(\\frac{(2k+1)\\pi}{2n}\\right)")

        for mob in [func, chebyshev_nodes_tex, Explained, chev_tex]:
            mob.set_color(BLACK)
            mob.set_stroke(BLACK, width=1)

        # --- Create axes ---
        axes = Axes(
            x_range=[-6,6,1],
            y_range=[-0.5,1.5,0.5],
            x_length=12,
            y_length=4.5,
            axis_config={"color": BLACK},
            tips=False
        ).scale(0.9)
        axes.move_to([0,-1.5,0])

        # --- Add axis labels ---
        x_labels, y_labels = axes.get_axis_labels(x_label="x", y_label="y")
        x_labels.set_color(GRAY)
        y_labels.set_color(GRAY)
        axes.add(x_labels, y_labels)
        axes.add_coordinates()
        for label in axes.x_axis.numbers + axes.y_axis.numbers:
            label.set_color(GRAY)

        # --- Plot the function ---
        graph = axes.plot(f, color=BLUE, stroke_width=5)

        # --- Equally spaced interpolation dots ---
        n_points = 11
        x_values = np.linspace(-5,5,n_points)
        y_values = f(x_values)
        dots = VGroup(*[Dot(axes.c2p(x,f(x)), color=PURPLE, radius=0.12) for x in x_values])

        # --- Lagrange interpolation for equally spaced points ---
        def Lag_int(x):
            total = 0
            n = len(x_values)
            for i in range(n):
                xi, yi = x_values[i], y_values[i]
                term = yi
                for j in range(n):
                    if j != i:
                        xj = x_values[j]
                        term *= (x - xj)/(xi - xj)
                total += term
            return total

        Lag_graph = axes.plot(Lag_int, color=RED_D, stroke_width=5)

        # --- Draw semicircle ---
        radius = 4.5
        semicircle_center = np.array([0,0.5,0])
        semicircle = Arc(radius=radius, start_angle=0, angle=PI, color=GREEN, stroke_width=5)

        # --- Chebyshev angles ---
        angles = [(2*k+1)/(2*n_points)*PI for k in range(n_points)]

        # --- Rays from semicircle center ---
        lines = VGroup()
        cx, cy = semicircle_center[0], semicircle_center[1]
        for ang in angles:
            x_end = cx + radius * np.cos(ang)
            y_end = cy + radius * np.sin(ang)
            lines.add(DashedLine(start=semicircle_center, end=np.array([x_end, y_end, 0]), color=PURE_RED))

        # --- Chebyshev nodes on x-axis ---
        cheb_x = [-5*chebyshev_node(k, n_points) for k in range(n_points)]

        # --- Vertical lines from semicircle endpoint to function ---
        vertical_lines = VGroup()

        factors = [0.58, 0.55, 0.53, 0.52, 0.52, 0.52, 0.52, 0.52, 0.53, 0.55, 0.58]

        for k, x in enumerate(cheb_x):
            theta = angles[k]

            y_end = 0
            end_point = axes.c2p(x, y_end)

            factor = factors[k]
            y_start = y_end + ((semicircle_center[1] + radius * np.sin(theta)) - y_end) * factor
            start_point = axes.c2p(x, y_start)

            vertical_lines.add(DashedLine(start=start_point, end=end_point, color=ORANGE))

        # --- Chebyshev nodes on function ---
        cheb_y = f(np.array(cheb_x))

        # --- Lagrange interpolation for Chebyshev nodes ---
        def Lag_int_chev(x):
            total = 0
            n = len(cheb_x)
            for i in range(n):
                xi, yi = cheb_x[i], cheb_y[i]
                term = yi
                for j in range(n):
                    if j != i:
                        xj = cheb_x[j]
                        term *= (x - xj)/(xi - xj)
                total += term
            return total

        Lag_graph_chev = axes.plot(Lag_int_chev, color=PURPLE_D, stroke_width=5)
        num_dots = 10
        dot_radius = 0.08
        radius_offset = -3  
        start_angle = PI/20  
        end_angle = PI - PI/20  

        angle_dots = VGroup()
        for i in range(num_dots):
            angle = start_angle + i * (end_angle - start_angle) / (num_dots - 1)
            x = semicircle_center[0] + (radius + radius_offset) * np.cos(angle)
            y = semicircle_center[1] + (radius + radius_offset) * np.sin(angle)- 2.1
            angle_dots.add(Dot([x, y, 0], radius=dot_radius, color=BLACK))

        move_animations = []
        for i, dot in enumerate(dots):
            target = axes.c2p(cheb_x[i], f(cheb_x[i]))
            move_animations.append(dot.animate.move_to(target))

        # --- Position formulas and semicircle ---
        func.move_to([0,1.5,0])
        chebyshev_nodes_tex.move_to([0,1.5,0])
        semicircle.move_to(semicircle_center)
        lines.move_to(semicircle_center)
        Explained.move_to([4.8,1.5,0])
        chev_tex.move_to([0,0,0])

        # --- Animations ---
        self.play(Write(chev_tex), run_time=3)
        self.wait(5.5)
        self.play(FadeOut(chev_tex), run_time=0.5)
        self.play(FadeIn(func, axes, Lag_graph, dots, graph), run_time=1)
        self.play(FadeOut(Lag_graph, func))
        self.play(FadeIn(chebyshev_nodes_tex), run_time=1)
        self.wait(0.5)
        self.play(Create(semicircle))
        self.play(Create(lines).set_run_time(3))
        self.wait(1)
        self.play(Create(VGroup(angle_dots, Explained)), run_time=1.5)
        self.play(Create(vertical_lines), run_time=3)
        self.play(*move_animations, run_time=3) 
        self.play(FadeOut(lines, vertical_lines, semicircle, Explained, angle_dots))
        self.play(Create(Lag_graph_chev), run_time=3)
        self.wait(5)
