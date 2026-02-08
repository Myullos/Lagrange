from manim import *
import numpy as np
from scipy.interpolate import lagrange

class ChebyshevSlopesAnimatedColored(MovingCameraScene):
    def construct(self):
        # background
        self.camera.background_color = WHITE
        explanation = Text(
            "Experimation 5:\n" \
            "座標点と傾きを用いた補間方法",
            font_size=36, color=BLACK, font="Hiragino Mincho ProN",line_spacing=1.5
            )
        explanation.to_edge(ORIGIN)
        self.play(FadeIn(explanation), run_time=0.5)
        self.wait(3.5)
        self.play(FadeOut(explanation), run_time=0.5)

        # Chebyshev 
        a, b = -5, 5
        n = 10
        k = np.arange(n+1)
        xs = 0.5*(a+b) + 0.5*(b-a)*np.cos((2*k+1)*np.pi/(2*(n+1)))
        xs = np.sort(xs)
        ys = 1 / (1 + xs**2)

        # initial slopes
        slopes = np.zeros(len(xs))
        for i in range(len(xs)):
            if i == 0:
                slopes[i] = (ys[1]-ys[0]) / (xs[1]-xs[0])
            elif i == len(xs)-1:
                slopes[i] = (ys[-1]-ys[-2]) / (xs[-1]-xs[-2])
            else:
                slopes[i] = (ys[i+1]-ys[i-1]) / (xs[i+1]-xs[i-1])

        # Axes
        y_max = ys.max()
        label_y = y_max + 1.2
        axes = Axes(
            x_range=[a-0.5, b+0.5, 1],
            y_range=[-0.5, y_max + 1.5, 0.5],
            tips=False
        ).scale(1.0)
        axes.shift(DOWN*0.5)
        axes.set_color(BLACK)
        self.add(axes)
        original_camera_center = self.camera.frame.get_center()
        original_camera_width = self.camera.frame.get_width()
        P_L = lagrange(xs, ys)  # Lagrange polynomial

        x_vals_L = np.linspace(a, b, 500)
        y_vals_L = np.array([float(P_L(x)) for x in x_vals_L])

        points_L = [axes.c2p(x, y) for x, y in zip(x_vals_L, y_vals_L)]

        lagrange_curve = VMobject()
        lagrange_curve.set_points_as_corners(points_L)
        lagrange_curve.set_color(ORANGE)
        lagrange_curve.set_stroke(width=3)

        # graph of true function
        graph_true = axes.plot(lambda x: 1/(1+x**2), color=BLUE_C)
        self.play(Create(graph_true), run_time=1.1)

        # animate each point
        arrow_length = 1.0
        arrow_stroke = 6
        label_offset = 0.3  # offset for label position
        def hermite_derivative_at(xs, ys, ms, i):
            # Endpoints use original slopes
            if i == 0 or i == len(xs)-1:
                return ms[i]

            hL = xs[i] - xs[i-1]

            yL, y0, yR = ys[i-1], ys[i], ys[i+1]
            mL, m0, mR = ms[i-1], ms[i], ms[i+1]

            # Left interval Cubic spline
            aL = (2*(yL - y0) + hL*(mL + m0)) / hL**3
            bL = (3*(y0 - yL) - hL*(2*mL + m0)) / hL**2

             # derivative (matching left and right)
            dL = (3*aL*hL + 2*bL)*hL + mL

            return dL

        all_labels = []
        all_lines = []
        all_green_arrows = []

        all_dots = []

        for i in range(len(xs)):
            p = axes.c2p(xs[i], ys[i])

            # 1. Interpolation point
            dot = Dot(p, color=RED)
            all_dots.append(dot)
            self.play(FadeIn(dot), run_time=0.05)

            # 2. tangent arrow
            vec = np.array([1, slopes[i]*1.5, 0])
            vec = vec / np.linalg.norm(vec) * arrow_length
            arrow = Arrow(p, p + vec, color=GREEN, buff=0, stroke_width=arrow_stroke, tip_length=0.3)
            all_green_arrows.append(arrow)


            # 3. label + dashed line
            label_pos = axes.c2p(xs[i], label_y) + UP*label_offset
            label = Tex(f"{slopes[i]:.2f}", font_size=20, color=BLACK)
            label.move_to(label_pos)

            # dashed line
            line = DashedLine(start=p, end=axes.c2p(xs[i], label_y), color=BLACK, dash_length=0.1, stroke_width=2)

            all_labels.append(label)
            # 4. slope expression
            if i == 0:
                expr = MathTex(
                    r"m_{0} = \frac{y_{1} - y_{0}}{x_{1} - x_{0}} = ",
                    f"{slopes[i]:.2f}"
                )
            elif i == len(xs) - 1:
                n_idx = len(xs) - 1
                expr = MathTex(
                    rf"m_{{{n_idx}}} = \frac{{y_{{{n_idx}}} - y_{{{n_idx-1}}}}}{{x_{{{n_idx}}} - x_{{{n_idx-1}}}}} = ",
                    f"{slopes[i]:.2f}"
                )
            else:
                expr = MathTex(
                    rf"m_{{{i}}} = \frac{{y_{{{i+1}}} - y_{{{i-1}}}}}{{x_{{{i+1}}} - x_{{{i-1}}}}} = ",
                    f"{slopes[i]:.2f}"
                )

            expr.set_color(BLACK)
            expr.to_edge(UP)

            created_line = Create(line)

            self.play(
                created_line,
                FadeIn(label),
                Create(arrow),
                FadeIn(expr),
                run_time=0.15
            )

            all_lines.append(created_line.mobject)

            self.wait(0.3)
            self.play(FadeOut(expr), run_time=0.15)

        mid = len(xs)//2 + 1 

        # axes' center point (absolute coordinates)
        p_left  = axes.c2p(xs[mid-1], ys[mid-1])
        p_right = axes.c2p(xs[mid+1], ys[mid+1])
        x_center_abs = (p_left[0] + p_right[0]) / 2
        y_center_abs = (p_left[1] + p_right[1]) / 2
        zoom_center = np.array([x_center_abs, y_center_abs, 0])

        self.play(
            FadeOut(VGroup(*all_labels, *all_lines)),
            run_time=0.3
        )

        self.play(
            self.camera.frame.animate.move_to(zoom_center).scale(0.5),
            run_time=0.5
        )

        arrow_length = 1.0
        arrow_stroke = 6

        # left and right slopes at mid point
        # left tangent
        left_slope = (ys[mid] - ys[mid-1]) / (xs[mid] - xs[mid-1])
        left_vec = np.array([1, left_slope*1.5, 0])
        left_vec = left_vec / np.linalg.norm(left_vec) * arrow_length
        left_arrow = Arrow(
            axes.c2p(xs[mid], ys[mid]),
            axes.c2p(xs[mid], ys[mid]) + left_vec,
            color=PURPLE, buff=0, stroke_width=arrow_stroke
        )

        # formula for left slope
        left_expr = MathTex(
            r"m_{\mathrm{left}} = \frac{y_i - y_{i-1}}{x_i - x_{i-1}}",
            color=PURPLE
        )

        left_expr.scale(0.6)

        left_expr.move_to(
            self.camera.frame.get_corner(UR) + LEFT*1.5 + DOWN*0.4
        )

        left_expr.set_z_index(10)
        left_expr.shift(LEFT*0.5 + DOWN*0.3)

        # right slope
        right_slope = (ys[mid+1] - ys[mid]) / (xs[mid+1] - xs[mid])
        right_vec = np.array([1, right_slope*1.5, 0])
        right_vec = right_vec / np.linalg.norm(right_vec) * arrow_length
        right_arrow = Arrow(
            axes.c2p(xs[mid], ys[mid]),
            axes.c2p(xs[mid], ys[mid]) + right_vec,
            color=RED, buff=0, stroke_width=arrow_stroke
        )

        right_expr = MathTex(
            r"m_{\mathrm{right}} = \frac{y_{i+1} - y_i}{x_{i+1} - x_i}",
            color=RED
        )

        # Match the size with left_expr
        right_expr.scale(0.6)

        # Position below left_expr as reference (remains correct after zoom)
        right_expr.next_to(left_expr, DOWN, aligned_edge=LEFT)
        right_expr.shift(RIGHT * 0.05)
        right_expr.set_z_index(10)

        p_mid = axes.c2p(xs[mid], ys[mid])
        p_left  = axes.c2p(xs[mid-1], ys[mid-1])
        p_right = axes.c2p(xs[mid+1], ys[mid+1])

        left_line = Line(p_left, p_mid, color=PURPLE, stroke_width=2)
        right_line = Line(p_mid, p_right, color=RED, stroke_width=2)

        # first, left side
        self.play(
            GrowArrow(left_arrow),
            FadeIn(left_expr),
            Create(left_line),   
            *[a.animate.set_opacity(0.2) for a in all_green_arrows],
            run_time=0.8
        )

        self.wait(0.6)

        # later, right side
        self.play(
            GrowArrow(right_arrow),
            FadeIn(right_expr),
            Create(right_line),   
            run_time=0.8
        )
        self.wait(1.2)

        # --- Slope required for cubic interpolation (purple) ---
        hermite_slope = hermite_derivative_at(xs, ys, slopes, mid)

        hermite_vec = np.array([1, hermite_slope*1.5, 0])
        hermite_vec = hermite_vec / np.linalg.norm(hermite_vec) * arrow_length

        hermite_arrow = Arrow(
            axes.c2p(xs[mid], ys[mid]),
            axes.c2p(xs[mid], ys[mid]) + hermite_vec,
            color=YELLOW,
            buff=0,
            stroke_width=arrow_stroke
        )

        hermite_expr = MathTex(
            r"m_i^{\mathrm{Hermite}} = H'(x_i)",
            color=YELLOW
        ).scale(0.6)

        hermite_expr.next_to(right_expr, DOWN, aligned_edge=LEFT)
        hermite_expr.set_z_index(10)

        self.play(
            GrowArrow(hermite_arrow),
            FadeIn(hermite_expr),
            run_time=0.8
        )

        self.wait(2)

        # 左区間 [x_{i-1}, x_i]
        hL = xs[mid] - xs[mid-1]
        yL, y0 = ys[mid-1], ys[mid]
        mL, m0 = slopes[mid-1], hermite_slope  # 左側傾き = Hermite で修正

        self.wait(1.5)

        # --- Hermite 説明フェーズ ---
        # カメラを元に戻す
        self.play(
            self.camera.frame.animate.move_to(original_camera_center).set(width=original_camera_width),
            run_time=1.0
        )

        # 全ての描画物をフェードアウト（補間点も含む）
        self.play(
            FadeOut(VGroup(
                graph_true, axes,
                left_line, right_line,
                left_arrow, right_arrow, hermite_arrow,
                *all_green_arrows,
                *all_dots,
                hermite_expr,
                left_expr, 
                right_expr
            )),
            run_time=1.0
        )
        self.wait(2.0)
        
        x_left = -3.5
        y_top = 2.0
        y_cursor = y_top

        # 条件式
        cond_texts = [
            r"H(x_{i-1}) = y_{i-1}",
            r"H(x_i) = y_i",
            r"H'(x_{i-1}) = m_{i-1}",
            r"H'(x_i) = m_i"
        ]
        cond_mobjects = []
        for txt in cond_texts:
            mobj = MathTex(txt, color=BLACK).scale(0.8)
            mobj.move_to([x_left, y_cursor, 0])
            y_cursor -= 0.8
            cond_mobjects.append(mobj)
            self.play(Write(mobj), run_time=0.4)
            self.wait(0.1)

        cond_group = VGroup(*cond_mobjects)
        highlight_box = SurroundingRectangle(cond_group, color=RED, stroke_width=3, buff=0.2)
        self.play(Create(highlight_box))
        self.wait(0.5)

        # --- H_L(x) 一般式 ---
        cubic_left = MathTex(
            r"H_L(x) = a_L(x-x_{i-1})^3 + b_L(x-x_{i-1})^2 + c_L(x-x_{i-1}) + d_L",
            color=BLACK
        ).scale(0.6).move_to([2.3,2.0 ,0])
        self.play(Write(cubic_left), run_time=1.0)
        self.wait(0.15)

        # --- H(x_{i-1}) = y_{i-1} 代入 ---
        sub1 = MathTex(
            r"H_L(x_{i-1}) = a_L (x_{i-1}-x_{i-1})^3 + b_L (x_{i-1}-x_{i-1})^2 \\"
            r"+ c_L (x_{i-1}-x_{i-1}) + d_L = y_{i-1}",
            color=BLACK
        ).scale(0.7).move_to([3,2.0 ,0])
        self.play(
            TransformMatchingTex(
                cubic_left, sub1,
                key_map={"a_L":"a_L","b_L":"b_L","c_L":"c_L","d_L":"d_L","x":"x_{i-1}"},
                transform_mismatches=True
            )
        )
        self.wait(0.15)

        # --- 簡略化: d_L = y_{i-1} ---
        simpl1 = MathTex(r"d_L = y_{i-1}", color=BLACK).scale(0.9).move_to([5,2.0 ,0])
        self.play(
            TransformMatchingTex(
                sub1, simpl1,
                key_map={"d_L":"d_L","y_{i-1}":"y_{i-1}"},
                transform_mismatches=True
            )
        )
        self.wait(0.15)

        # --- H'(x) の一般式 ---
        deriv_expr = MathTex(
            r"H_L'(x) = 3 a_L (x-x_{i-1})^2 + 2 b_L (x-x_{i-1}) + c_L",
            color=BLACK
        ).scale(0.7).move_to([2.3,1.4 ,0])
        self.play(Write(deriv_expr), run_time=1.0)
        self.wait(0.15)

        # --- H'(x_{i-1}) = m_{i-1} 代入 ---
        sub2 = MathTex(
            r"H_L'(x_{i-1}) = 3 a_L (x_{i-1}-x_{i-1})^2 + \\"
            r"2 b_L (x_{i-1}-x_{i-1}) + c_L = m_{i-1}",
            color=BLACK
        ).scale(0.8).move_to([3.2,1.2 ,0])
        self.play(
            TransformMatchingTex(
                deriv_expr, sub2,
                key_map={"a_L":"a_L","b_L":"b_L","c_L":"c_L","x":"x_{i-1}","m_{i-1}":"m_{i-1}"},
                transform_mismatches=True
            )
        )
        self.wait(0.15)

        # --- 簡略化: c_L = m_{i-1} ---
        simpl2 = MathTex(r"c_L = m_{i-1}", color=BLACK).scale(0.9).move_to([5,1.4 ,0])
        self.play(
            TransformMatchingTex(
                sub2, simpl2,
                key_map={"c_L":"c_L","m_{i-1}":"m_{i-1}"},
                transform_mismatches=True
            )
        )
        self.wait(0.15)

        # --- H(x_i) = y_i 代入 ---
        sub3 = MathTex(
            r"H_L(x_i) = a_L (x_i-x_{i-1})^3 + b_L (x_i-x_{i-1})^2 + \\"
            r"c_L (x_i-x_{i-1}) + d_L = y_i",
            color=BLACK
        ).scale(0.7).move_to([2.3,0.6 ,0])
        self.play(Write(sub3))
        self.wait(0.15)

        # --- d_L, c_L を代入して簡略化 ---
        simpl3 = MathTex(
            r"a_L (x_i-x_{i-1})^3 + b_L (x_i-x_{i-1})^2+ \\"
            r"+ m_{i-1} (x_i-x_{i-1}) + y_{i-1} = y_i",
            color=BLACK
        ).scale(0.8).move_to([3,0.5 ,0])
        self.play(
            TransformMatchingTex(
                sub3, simpl3,
                key_map={"a_L":"a_L","b_L":"b_L","m_{i-1}":"m_{i-1}","y_{i-1}":"y_{i-1}","y_i":"y_i"},
                transform_mismatches=True
            )
        )
        self.wait(0.15)

        a_calc = MathTex(
            r"a_L = \frac{m_i - m_{i-1} - 2 b_L (x_i-x_{i-1})}{3 (x_i-x_{i-1})^2}",
            color=BLACK
        ).scale(0.8).move_to([2.5, -0.4, 0])  # 横に並べる

        b_calc = MathTex(
            r"b_L = \frac{y_i - y_{i-1} - m_{i-1} (x_i-x_{i-1}) - a_L (x_i-x_{i-1})^3}{(x_i-x_{i-1})^2}",
            color=BLACK
        ).scale(0.7).move_to([2.1, 0.6, 0])  # 横は同じ、y は少し下

        # simpl3 から a_calc と b_calc に同時変形
        self.play(
            TransformMatchingTex(simpl3, a_calc, key_map={"a_L":"a_L","b_L":"b_L","m_{i-1}":"m_{i-1}","y_{i-1}":"y_{i-1}","y_i":"y_i"}, transform_mismatches=True),
            TransformMatchingTex(simpl3, b_calc, key_map={"a_L":"a_L","b_L":"b_L","m_{i-1}":"m_{i-1}","y_{i-1}":"y_{i-1}","y_i":"y_i"}, transform_mismatches=True)
        )
        self.wait(0.15)
        slope_explanation_text = Text(
            "初期傾き m_i は左右の傾き m_L, m_R の平均で設定",
            font_size=28, color=BLACK, font="Hiragino Mincho ProN"
        )
        slope_explanation_text.move_to([0, -1.2, 0])

        slope_explanation_formula = MathTex(
            r"m_i = \frac{m_L + m_R}{2}",
            color=BLACK
        ).scale(1.0)
        slope_explanation_formula.next_to(slope_explanation_text, DOWN, buff=0.3)

        self.play(FadeIn(slope_explanation_text, shift=DOWN))
        self.play(Write(slope_explanation_formula))
        self.wait(1.0)

        # --- 実際の数値は差分法で計算 ---
        mid = len(xs)//2 + 1
        mL = slopes[mid-1]
        mR = slopes[mid+1]
        m0 = hermite_derivative_at(xs, ys, slopes, mid)  # ここでは元の方法で計算
        yL, y0 = ys[mid-1], ys[mid]
        hL = xs[mid] - xs[mid-1]

        aL_val = (2*(yL - y0) + hL*(mL + m0)) / hL**3
        bL_val = (3*(y0 - yL) - hL*(2*mL + m0)) / hL**2
        cL_val = mL
        dL_val = yL
        x_left_val = xs[mid-1]

        # --- 一般式と数値代入の表示 ---
        cubic_left_general = MathTex(
            r"H_L(x) = a_L (x-x_{i-1})^3 + b_L (x-x_{i-1})^2 + c_L (x-x_{i-1}) + d_L",
            color=BLACK
        ).scale(0.8).move_to([0, -3.4, 0])
        self.play(Write(cubic_left_general))
        self.wait(0.5)

        cubic_left_numeric = MathTex(
            rf"H_L(x) = {aL_val:.3f} (x - {x_left_val:.3f})^3 + {bL_val:.3f} (x - {x_left_val:.3f})^2 "
            rf"+ {cL_val:.3f} (x - {x_left_val:.3f}) + {dL_val:.3f}",
            color=BLACK
        ).scale(0.8).move_to([0, -3.4, 0])
        self.play(TransformMatchingTex(cubic_left_general, cubic_left_numeric))
        self.wait(1.0)
        # --- ここまでに出てきたものを全て消す ---
        self.play(
            FadeOut(VGroup(
                # 条件式
                *cond_mobjects,
                highlight_box,
                # 一般式・代入・簡略化式
                simpl1, simpl2, a_calc, b_calc,
                # 傾きの説明
                slope_explanation_text, slope_explanation_formula,
                # 数値代入版
                cubic_left_numeric
            )),
            run_time=0.4
        )
        # --- カメラを左上に移動 ---
        self.play(
            self.camera.frame.animate.move_to(left_expr.get_center()).set(width=original_camera_width*0.6),
            run_time=0.05
        )

            # --- 必要なものだけ FadeIn ---
        self.play(
            FadeIn(graph_true),
            *[FadeIn(dot) for dot in all_dots],
            FadeIn(axes),
            FadeIn(hermite_arrow),
            run_time=0.3
        )

        # --- 全区間の三次補間を描画 ---
        hermite_lines = VGroup()
        for i in range(len(xs)-1):
            h = xs[i+1] - xs[i]
            y0, y1 = ys[i], ys[i+1]
            m0 = slopes[i]
            m1 = hermite_derivative_at(xs, ys, slopes, i+1)

            a = (2*(y0 - y1) + h*(m0 + m1)) / h**3
            b = (3*(y1 - y0) - h*(2*m0 + m1)) / h**2
            c = m0
            d = y0

            x_vals = np.linspace(xs[i], xs[i+1], 50)
            y_vals = a*(x_vals - xs[i])**3 + b*(x_vals - xs[i])**2 + c*(x_vals - xs[i]) + d

            points = [axes.c2p(x, y) for x, y in zip(x_vals, y_vals)]
            curve = VMobject()
            curve.set_points_smoothly(points)
            curve.set_color(GREEN)
            hermite_lines.add(curve)
        self.play(Create(hermite_lines), run_time=1.0)
            
        self.play(
        self.camera.frame.animate.move_to(original_camera_center).set(width=original_camera_width),
        run_time=1
        )

        # Lagrange 多項式を作る
        self.play(FadeIn(lagrange_curve), FadeOut(hermite_arrow),run_time=1)
        self.wait(5)

