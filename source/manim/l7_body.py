# Lesson 7 · MSE as Gaussian MLE → gradient descent → logistic regression (slow part 2 of the Lesson 4 split)
C_TRUE = "#7aa2f7"; C_PRED = "#ff9e64"; C_ERR = "#e0af68"; C_W = "#b392f0"; C_ETA = "#5fc9b6"
C_LOSS = "#e08fd0"; C_SIGM = "#7dcfff"; C_KL7 = "#f7768e"; C_H7 = "#9ece6a"

def rows7(*items, buff=0.3, size=28):
    out = VGroup()
    for it in items:
        if isinstance(it, tuple): out.add(Text(it[0], font_size=it[2] if len(it) > 2 else size, color=it[1]))
        else: out.add(Text(it, font_size=size))
    return out.arrange(DOWN, aligned_edge=LEFT, buff=buff)

def atl(g, x=-0.6):
    if g.width > 6.95 - x: g.scale_to_fit_width(6.95 - x)
    g.shift(RIGHT * (x - g.get_left()[0])); return g

def bars7(ps, labels, color, w=0.8, gap=0.35, hmax=1.6, fs=22, title=None):
    g = VGroup(); bars = VGroup()
    for i, (p, l) in enumerate(zip(ps, labels)):
        x = i * (w + gap); v = eval(p.replace("½", "0.5")) if p not in ("0", "1") else float(p); h = max(0.03, v * hmax)
        r = Rectangle(width=w, height=h, fill_color=color, fill_opacity=0.85, stroke_width=0).move_to([x, h / 2, 0])
        val = Text(p, font_size=fs, color=FG).next_to(r, UP, buff=0.08)
        lab = Text(l, font_size=fs, color=MUTED).move_to([x, -0.28, 0])
        base = Line([x - w / 2 - 0.08, 0, 0], [x + w / 2 + 0.08, 0, 0], color=GRID, stroke_width=2)
        bars.add(VGroup(base, r, val, lab))
    g.add(bars)
    if title: g.add(Text(title, font_size=fs + 2, color=color).next_to(bars, UP, buff=0.35))
    return g

# ---------------------------------------------------------------- Ch1: Recall
class Recall7(StepScene):
    def construct(self):
        title = caption("Recall first · four ideas from earlier lessons", 32).to_edge(UP, buff=0.35)
        # 1 CE = H + KL
        self.begin("cekl")
        self.play(FadeIn(title), run_time=0.7)
        h = Text("Lesson 6 · cross-entropy = entropy + KL", font_size=28, color=C_LOSS).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(h), run_time=0.7)
        e = MathTex(r"\underbrace{H(P,Q)}_{2\text{ bits}}", r"=", r"\underbrace{H(P)}_{1.5\text{ bits}}", r"+", r"\underbrace{D_{KL}(P\|Q)}_{?}", font_size=50).shift(UP * 0.4)
        e[0].set_color(C_LOSS); e[2].set_color(C_H7); e[4].set_color(C_KL7)
        self.play(Write(e), run_time=1.6)
        a = Text("D_KL = 2 − 1.5 = 0.5 bits", font_size=30, color=C_KL7).shift(DOWN * 1.1)
        self.play(FadeIn(a), run_time=0.8)
        b = Text("KL ≥ 0, so no model's cross-entropy can go below H(P) = 1.5", font_size=24, color=MUTED).shift(DOWN * 1.9)
        self.play(FadeIn(b), run_time=0.8)
        self.say("training can only shrink the KL part; H(P) is fixed by the data", FG, 26)
        self.end()
        # 2 KL asymmetry
        self.begin("asym")
        self.play(FadeOut(VGroup(h, e, a, b)), run_time=0.5)
        h2 = Text("Lesson 6 · KL is not symmetric", font_size=28, color=C_KL7).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(h2), run_time=0.6)
        P = bars7(["½", "½"], ["a", "b"], C_TRUE, title="P", w=0.9).shift(LEFT * 5.6 + DOWN * 0.6)
        Q = bars7(["1", "0"], ["a", "b"], C_PRED, title="Q", w=0.9).shift(LEFT * 3.0 + DOWN * 0.6)
        self.play(FadeIn(P), FadeIn(Q), run_time=0.9)
        a1 = MathTex(r"D_{KL}(Q\|P)=1\cdot\log_2\tfrac{1}{1/2}+0=1\text{ bit}", font_size=36, color=C_KL7).shift(RIGHT * 2.4 + UP * 0.7)
        a2 = MathTex(r"D_{KL}(P\|Q)=\tfrac12\log_2\tfrac{1/2}{1}+\tfrac12\log_2\tfrac{1/2}{0}=\infty", font_size=36, color=C_BAD).shift(RIGHT * 2.4 + DOWN * 0.6)
        self.play(Write(a1), run_time=1.3); self.wait(0.4)
        self.play(Write(a2), run_time=1.5)
        self.say("∞ whenever the second distribution gives 0 to something the first allows", C_BAD, 26)
        self.end()
        # 3 AUROC with a tie
        self.begin("auroc")
        self.play(FadeOut(VGroup(h2, P, Q, a1, a2)), run_time=0.5)
        h3 = Text("Lesson 3 · AUROC by counting pairs, with a tie", font_size=28, color=C_GOOD).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(h3), run_time=0.6)
        sc = Text("positives: 0.9, 0.6      negatives: 0.6, 0.2", font_size=28, color=FG).next_to(h3, DOWN, buff=0.35)
        self.play(FadeIn(sc), run_time=0.7)
        pairs = [("0.9 vs 0.6", "pos higher", "1", C_GOOD), ("0.9 vs 0.2", "pos higher", "1", C_GOOD),
                 ("0.6 vs 0.6", "tie", "½", C_ERR), ("0.6 vs 0.2", "pos higher", "1", C_GOOD)]
        tbl = VGroup()
        for i, (pr, w_, v, c) in enumerate(pairs):
            r = VGroup(Text(pr, font_size=26), Text(w_, font_size=24, color=c), Text(v, font_size=28, color=c))
            r[0].move_to([-3.0, 0, 0]); r[1].move_to([0.2, 0, 0]); r[2].move_to([2.6, 0, 0])
            r.shift(DOWN * (i * 0.55))
            tbl.add(r)
        tbl.move_to(DOWN * 0.55)
        for r in tbl: self.play(FadeIn(r), run_time=0.55); self.wait(0.2)
        tot = Text("AUROC = (1 + 1 + ½ + 1) / 4 = 3.5 / 4 = 0.875", font_size=28, color=C_GOOD).shift(DOWN * 2.35)
        self.play(FadeIn(tot), run_time=0.8)
        self.say("a tie counts as half a win: every positive × negative pair, once", FG, 26)
        self.end()
        # 4 PPO clip inside the range
        self.begin("clip")
        self.play(FadeOut(VGroup(h3, sc, tbl, tot)), run_time=0.5)
        h4 = Text("Lesson 1 · PPO, good response (A > 0) · ρ = 1.1 · ε = 0.2", font_size=28, color=C_RHO).next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(h4), run_time=0.7)
        ax = Axes(x_range=[0.5, 1.6, 0.1], y_range=[0, 1.5, 0.5], x_length=7.5, y_length=3.0,
                  axis_config={"color": MUTED, "stroke_width": 2}, tips=False).shift(DOWN * 0.6)
        xl = Text("ρ = π_θ / π_old", font_size=22, color=MUTED).next_to(ax, DOWN, buff=0.12)
        yl = Text("objective", font_size=22, color=MUTED).next_to(ax, LEFT, buff=0.12)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=1.0)
        f = ax.plot(lambda r: min(r, 1.2), x_range=[0.5, 1.6], color=C_PI, stroke_width=5)
        self.play(Create(f), run_time=1.3)
        e = DashedLine(ax.c2p(1.2, 0), ax.c2p(1.2, 1.45), color=C_EPS, stroke_width=3)
        el = Text("1 + ε = 1.2", font_size=22, color=C_EPS).next_to(e, UP, buff=0.08)
        dot = Dot(ax.c2p(1.1, 1.1), color=C_ERR, radius=0.1)
        dl = Text("ρ = 1.1: still rising → gradient ≠ 0", font_size=24, color=C_ERR).next_to(dot, DR, buff=0.15)
        self.play(Create(e), FadeIn(el), run_time=0.8)
        self.play(FadeIn(dot), FadeIn(dl), run_time=0.8)
        self.say("inside [1 − ε, 1 + ε] nothing is clipped: it keeps getting pushed up to 1.2", C_ERR, 24)
        self.end(2.2)

# ---------------------------------------------------------------- Ch2: MSE from Gaussian noise
def gauss(x, s=1.0): return np.exp(-x * x / (2 * s * s))

class GaussMSE(StepScene):
    def construct(self):
        title = caption("Why squared error? Because of the noise you assume", 32).to_edge(UP, buff=0.35)
        # 1 the story: truth = prediction + noise
        self.begin("noise")
        self.play(FadeIn(title), run_time=0.7)
        ax = Axes(x_range=[0, 6, 1], y_range=[0, 6, 1], x_length=5.6, y_length=4.2,
                  axis_config={"color": MUTED, "stroke_width": 2}, tips=False).shift(LEFT * 3.2 + DOWN * 0.45)
        xl = Text("clip length", font_size=20, color=MUTED).next_to(ax, DOWN, buff=0.1)
        yl = Text("seconds to transcribe", font_size=20, color=MUTED).rotate(PI / 2).next_to(ax, LEFT, buff=0.1)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=1.0)
        line = ax.plot(lambda x: 0.8 * x + 0.6, x_range=[0, 6], color=C_PRED, stroke_width=4)
        ll = Text("prediction ŷ", font_size=22, color=C_PRED).next_to(ax.c2p(5.6, 5.08), UP, buff=0.12)
        self.play(Create(line), FadeIn(ll), run_time=1.0)
        xs = [0.8, 1.6, 2.4, 3.1, 3.9, 4.6, 5.3]; ns = [0.25, -0.35, 0.1, 0.55, -0.2, -0.6, 0.3]
        dots = VGroup(*[Dot(ax.c2p(x, 0.8 * x + 0.6 + n), color=C_TRUE, radius=0.07) for x, n in zip(xs, ns)])
        segs = VGroup(*[Line(ax.c2p(x, 0.8 * x + 0.6), ax.c2p(x, 0.8 * x + 0.6 + n), color=C_ERR, stroke_width=3) for x, n in zip(xs, ns)])
        self.play(FadeIn(dots), run_time=0.9)
        self.play(Create(segs), run_time=0.9)
        side = rows7(("truth y = prediction ŷ + noise", FG, 28), ("noise = the miss y − ŷ", C_ERR, 26),
                     ("small misses: common", C_GOOD, 26), ("big misses: rare", C_BAD, 26), buff=0.35).shift(RIGHT * 3.3 + UP * 0.4)
        for it in side: self.play(FadeIn(it, shift=RIGHT * 0.2), run_time=0.7); self.wait(0.3)
        self.say("to score a model, ask: how likely was each real y under it?", FG, 28)
        self.end()
        # 2 the bell curve
        self.begin("bell")
        self.play(FadeOut(VGroup(ax, xl, yl, line, ll, dots, segs, side)), run_time=0.5)
        bx = Axes(x_range=[-3.5, 3.5, 1], y_range=[0, 1.1, 0.5], x_length=6.4, y_length=2.6,
                  axis_config={"color": MUTED, "stroke_width": 2}, tips=False).shift(DOWN * 1.1 + LEFT * 2.9)
        bl = Text("miss  y − ŷ  (σ = 1)", font_size=22, color=C_ERR).next_to(bx, DOWN, buff=0.12)
        bell = bx.plot(lambda x: gauss(x), x_range=[-3.5, 3.5], color=C_TRUE, stroke_width=5)
        f = MathTex(r"p(y\mid\hat y)=", r"\frac{1}{\sqrt{2\pi\sigma^2}}", r"\,e^{-\frac{(y-\hat y)^2}{2\sigma^2}}", font_size=48).next_to(title, DOWN, buff=0.4)
        f[2].set_color(C_TRUE)
        self.play(Write(f), run_time=1.6)
        self.play(Create(bx), FadeIn(bl), Create(bell), run_time=1.3)
        notes = rows7(("e^(− squared distance): the bell shape", C_TRUE, 24), ("σ = how wide the noise is", FG, 24),
                      ("front factor only makes the area = 1", MUTED, 24), ("miss 1σ → height 0.61, 3σ → 0.01", C_ERR, 24), buff=0.32).shift(RIGHT * 3.6 + DOWN * 0.9)
        for it in notes: self.play(FadeIn(it), run_time=0.6); self.wait(0.3)
        self.say("the Gaussian (bell curve) = “misses get rare fast as they grow”", FG, 26)
        self.end()
        # 3 surprise = -log p
        self.begin("neglog")
        self.play(FadeOut(VGroup(bx, bl, bell, notes, f)), run_time=0.5)
        s = MathTex(r"-\ln p(y\mid\hat y)=", r"\frac{(y-\hat y)^2}{2\sigma^2}", r"+\ln\sqrt{2\pi\sigma^2}", font_size=46).shift(UP * 0.55)
        s[1].set_color(C_ERR); s[2].set_color(MUTED)
        hint = Text("Lesson 6: loss = surprise = −log p.   ln cancels e:  −ln e^(−a) = a", font_size=24, color=C_LOSS).next_to(title, DOWN, buff=0.45)
        self.play(FadeIn(hint), run_time=0.7)
        self.play(Write(s), run_time=1.6)
        tb = VGroup()
        for i, (m, v) in enumerate([("miss 1", "1²/2 = 0.5"), ("miss 2", "2²/2 = 2"), ("miss 3", "3²/2 = 4.5")]):
            r = VGroup(Text(m, font_size=26, color=C_ERR), Text(v + "  + const", font_size=26))
            r[0].move_to([-2.4, 0, 0]); r[1].move_to([1.0, 0, 0], aligned_edge=LEFT).shift(LEFT * 0.4)
            tb.add(r.shift(DOWN * i * 0.5))
        tb.move_to(DOWN * 1.4)
        for r in tb: self.play(FadeIn(r), run_time=0.55); self.wait(0.2)
        self.say("worked example 1 (σ = 1): double the miss → 4× the surprise", C_ERR, 26)
        self.end()
        # 4 many points: products → sums → MSE
        self.begin("sum")
        self.play(FadeOut(VGroup(hint, s, tb)), run_time=0.5)
        l1 = MathTex(r"p(\text{all } y)=\prod_i p(y_i\mid\hat y_i)", font_size=42).next_to(title, DOWN, buff=0.45)
        l2 = MathTex(r"-\ln p(\text{all } y)=\sum_i \frac{(y_i-\hat y_i)^2}{2\sigma^2}+N\ln\sqrt{2\pi\sigma^2}", font_size=42).next_to(l1, DOWN, buff=0.35)
        l2[0][15:].set_color(C_ERR)
        self.play(Write(l1), run_time=1.2)
        n1 = Text("independent points: probabilities multiply → surprises add (Lesson 6)", font_size=22, color=MUTED).next_to(l2, DOWN, buff=0.12)
        self.play(Write(l2), FadeIn(n1), run_time=1.6)
        l3 = MathTex(r"\text{MSE}=\frac1N\sum_i (y_i-\hat y_i)^2", font_size=50, color=C_LOSS).shift(DOWN * 1.75)
        n2 = Text("σ and the constant don't depend on the model → drop them", font_size=24, color=FG).next_to(l3, UP, buff=0.25)
        self.play(FadeIn(n2), run_time=0.7)
        self.play(Write(l3), run_time=1.2)
        self.say("max likelihood under Gaussian noise = min MSE", C_LOSS, 28)
        self.end()
        # 5 edge case: an outlier + faded
        self.begin("edge")
        self.play(FadeOut(VGroup(l1, l2, n1, n2, l3)), run_time=0.5)
        A = rows7(("model A · misses 1, 1, 1", C_PRED, 28), ("MSE = (1 + 1 + 1)/3 = 1", FG, 28), buff=0.25).shift(LEFT * 3.3 + UP * 1.2)
        B = rows7(("model B · misses 0, 0, 3", C_PRED, 28), ("MSE = (0 + 0 + 9)/3 = 3", C_BAD, 28), buff=0.25).shift(RIGHT * 3.0 + UP * 1.2)
        self.play(FadeIn(A), run_time=0.8); self.wait(0.3)
        self.play(FadeIn(B), run_time=0.8)
        n = rows7(("same total miss (3), but B costs 3× more", FG, 26),
                  ("the bell says a 3σ miss is nearly impossible → one outlier dominates", MUTED, 22),
                  ("lots of outliers? |y − ŷ| (MAE) = heavier-tailed (Laplace) noise", C_ETA, 22), buff=0.28).shift(DOWN * 0.45)
        for it in n: self.play(FadeIn(it), run_time=0.6); self.wait(0.3)
        fd = Text("your turn: misses 2, 0, −2 → MSE = ?", font_size=28, color=C_AL).shift(DOWN * 2.1)
        q = SurroundingRectangle(fd, color=C_AL, buff=0.12)
        self.play(FadeIn(fd), Create(q), run_time=0.8)
        self.say("worked example 2 (edge case): squaring punishes one big miss", C_BAD, 26)
        self.end(2.2)

# ---------------------------------------------------------------- Ch3: Gradient descent
L3f = lambda w: (w - 3) ** 2

class GradDesc7(StepScene):
    def bowl(self):
        ax = Axes(x_range=[-0.5, 6.5, 1], y_range=[0, 12.5, 4], x_length=5.6, y_length=4.3,
                  axis_config={"color": MUTED, "stroke_width": 2, "include_numbers": True, "font_size": 20}, tips=False).shift(LEFT * 3.95 + DOWN * 0.55)
        cur = ax.plot(L3f, x_range=[-0.46, 6.46], color=C_LOSS, stroke_width=5)
        lab = MathTex(r"L(w)=(w-3)^2", font_size=34, color=C_LOSS).next_to(ax.c2p(5.4, 11.5), LEFT, buff=0.1)
        wl = Text("w", font_size=24, color=C_W).next_to(ax, RIGHT, buff=0.1).shift(DOWN * 2.0)
        return ax, cur, lab, wl
    def construct(self):
        title = caption("Gradient descent: walk downhill in the fog", 32).to_edge(UP, buff=0.35)
        # 1 the hill and the slope
        self.begin("hill")
        self.play(FadeIn(title), run_time=0.7)
        ax, cur, lab, wl = self.bowl()
        self.play(Create(ax), FadeIn(wl), run_time=1.0)
        self.play(Create(cur), FadeIn(lab), run_time=1.2)
        ball = Dot(ax.c2p(5, 4), color=C_W, radius=0.12)
        tan = ax.plot(lambda w: 4 + 4 * (w - 5), x_range=[4.2, 5.8], color=C_ERR, stroke_width=4)
        self.play(FadeIn(ball), run_time=0.6)
        self.play(Create(tan), run_time=0.8)
        side = rows7(("= squared error of ŷ = w·x on one point x = 1, y = 3", MUTED, 20),
                     ("nudge test: L(5) = 4,  L(5.1) = 4.41", FG, 24),
                     ("slope ≈ 0.41 / 0.1 ≈ 4", C_ERR, 24), buff=0.3).shift(UP * 0.9); atl(side)
        for it in side: self.play(FadeIn(it), run_time=0.6); self.wait(0.3)
        d = MathTex(r"\frac{dL}{dw}=2(w-3)=2\cdot2=4", font_size=40, color=C_ERR).next_to(side, DOWN, buff=0.45)
        dn = Text("chain rule: (u²)′ = 2u · u′ with u = w − 3", font_size=20, color=MUTED).next_to(d, DOWN, buff=0.2)
        self.play(Write(d), FadeIn(dn), run_time=1.2)
        self.say("the slope (derivative) tells you which way is uphill, and how steep", FG, 26)
        self.end()
        # 2 the update rule
        self.begin("rule")
        self.play(FadeOut(VGroup(side, d, dn)), run_time=0.5)
        r = MathTex(r"w", r"\leftarrow", r"w", r"-", r"\eta", r"\cdot", r"\frac{dL}{dw}", font_size=56).shift(UP * 1.3); atl(r, 0.4)
        r[0].set_color(C_W); r[2].set_color(C_W); r[4].set_color(C_ETA); r[6].set_color(C_ERR)
        self.play(Write(r), run_time=1.4)
        ex = rows7(("slope > 0 → uphill is to the right", C_ERR, 24), ("minus sign → step left, downhill", FG, 24),
                   ("η (learning rate) = step per unit of slope", C_ETA, 24), ("steep → far from the bottom → big step", MUTED, 22), buff=0.3).next_to(r, DOWN, buff=0.45); atl(ex)
        for it in ex: self.play(FadeIn(it), run_time=0.6); self.wait(0.3)
        arr = Arrow(ax.c2p(5, 4) + UP * 0.35, ax.c2p(4, 1) + UP * 0.35 + RIGHT * 0.1, color=C_W, buff=0.05, stroke_width=5)
        self.play(GrowArrow(arr), run_time=0.8)
        self.say("step against the slope, scaled by the learning rate η", C_ETA, 28)
        self.end()
        # 3 worked example 1: eta = 0.25
        self.begin("ex1")
        self.play(FadeOut(VGroup(r, ex, arr, tan)), run_time=0.5)
        hdr = atl(Text("worked example 1 · η = 0.25 · start w = 5", font_size=26, color=C_ETA).shift(UP * 2.0))
        self.play(FadeIn(hdr), run_time=0.6)
        steps = [(5, 4, 4), (4, 2, 3.5), (3.5, 1, 3.25)]
        lines = VGroup()
        for i, (w0, g, w1) in enumerate(steps):
            t = Text(f"w = {w0:g}:  slope {g:g}  →  w = {w0:g} − 0.25·{g:g} = {w1:g}", font_size=22, color=FG)
            lines.add(t)
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(hdr, DOWN, buff=0.35); atl(lines)
        for (w0, g, w1), t in zip(steps, lines):
            self.play(FadeIn(t), run_time=0.6)
            nb = Dot(ax.c2p(w1, L3f(w1)), color=C_W, radius=0.12)
            self.play(ball.animate.move_to(nb), run_time=0.9); self.wait(0.3)
        why = rows7(("distance to 3:  2 → 1 → 0.5 → 0.25", C_GOOD, 24),
                    ("why: w − 3  ←  (1 − 2η)(w − 3) = 0.5·(w − 3)", MUTED, 22), buff=0.28).next_to(lines, DOWN, buff=0.4, aligned_edge=LEFT)
        self.play(FadeIn(why), run_time=0.8)
        self.say("each step halves the distance to the bottom: it converges", C_GOOD, 26)
        self.end()
        # 4 worked example 2: too big a learning rate
        self.begin("ex2")
        self.play(FadeOut(VGroup(hdr, lines, why)), ball.animate.move_to(ax.c2p(5, 4)), run_time=0.7)
        hdr2 = atl(Text("worked example 2 (edge case) · η = 1.1", font_size=26, color=C_BAD).shift(UP * 2.0))
        self.play(FadeIn(hdr2), run_time=0.6)
        st2 = [(5, 4, 0.6), (0.6, -4.8, 5.88)]
        l2 = VGroup(Text("w = 5:  slope 4  →  5 − 1.1·4 = 0.6", font_size=22),
                    Text("w = 0.6:  slope −4.8  →  0.6 + 5.28 = 5.88", font_size=22)).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(hdr2, DOWN, buff=0.35); atl(l2)
        prev = ax.c2p(5, 4)
        for (w0, g, w1), t in zip(st2, l2):
            self.play(FadeIn(t), run_time=0.6)
            p1 = ax.c2p(w1, L3f(w1))
            self.play(Create(Arrow(prev, p1, color=C_BAD, buff=0.1, stroke_width=4)), ball.animate.move_to(p1), run_time=1.0)
            prev = p1; self.wait(0.3)
        w2 = rows7(("distance to 3:  2 → 2.4 → 2.88  (×1.2 per step)", C_BAD, 22),
                   ("1 − 2η = −1.2: flips side and grows → diverges", MUTED, 22),
                   ("η = 1 bounces 5, 1, 5, 1 … forever", FG, 22), ("in code: loss climbs, then NaN", C_BAD, 22), buff=0.28).next_to(l2, DOWN, buff=0.4, aligned_edge=LEFT)
        for it in w2: self.play(FadeIn(it), run_time=0.6); self.wait(0.25)
        self.say("too big a learning rate overshoots the bottom every step", C_BAD, 26)
        self.end()
        # 5 faded + the real thing
        self.begin("faded")
        self.play(FadeOut(VGroup(*[m for m in self.mobjects if isinstance(m, Arrow)], hdr2, l2, w2)), ball.animate.move_to(ax.c2p(5, 4)), run_time=0.7)
        fd = atl(rows7(("your turn · η = 0.5 · start w = 5", C_AL, 26), ("slope 4  →  w = 5 − 0.5·4 = ?", C_AL, 26), buff=0.3).shift(UP * 1.5), 0.6)
        q = SurroundingRectangle(fd, color=C_AL, buff=0.15)
        self.play(FadeIn(fd), Create(q), run_time=0.9)
        real = rows7(("the real thing: millions of weights, not one", FG, 24),
                     ("∇L = one slope per weight (the gradient)", C_ERR, 22),
                     ("your GRPO run: lr = 1e-5 is this η", C_ETA, 22), buff=0.3).next_to(q, DOWN, buff=0.55); atl(real)
        for it in real: self.play(FadeIn(it), run_time=0.6); self.wait(0.3)
        self.say("same rule for every weight: w ← w − η · ∇L", C_ETA, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch4: Logistic regression
def sgm(z): return 1 / (1 + np.exp(-z))

class LogReg7(StepScene):
    def construct(self):
        title = caption("Logistic regression: four pieces you already have", 32).to_edge(UP, buff=0.35)
        # 1 recap bridge
        self.begin("recap")
        self.play(FadeIn(title), run_time=0.7)
        cards = VGroup()
        for t, f, c in [("score", r"z=w\cdot x", C_W), ("probability", r"p=\;?", C_SIGM), ("loss = surprise", r"-\ln(\text{prob of true label})", C_LOSS), ("step", r"w\leftarrow w-\eta\,\tfrac{dL}{dw}", C_ETA)]:
            box = RoundedRectangle(width=3.2, height=1.9, corner_radius=0.15, stroke_color=c, fill_color=c, fill_opacity=0.08)
            g = VGroup(Text(t, font_size=24, color=c), MathTex(f, font_size=30, color=FG)).arrange(DOWN, buff=0.3).move_to(box)
            if g.width > 3.0: g.scale_to_fit_width(3.0)
            cards.add(VGroup(box, g))
        cards.arrange(RIGHT, buff=0.22).shift(UP * 0.4)
        srcs = VGroup(*[Text(s, font_size=20, color=MUTED).next_to(cd, DOWN, buff=0.2) for s, cd in zip(["Ch 3's ŷ = w·x", "NEW today", "Lesson 6", "Ch 3"], cards)])
        srcs[1].set_color(C_SIGM)
        for cd, s in zip(cards, srcs): self.play(FadeIn(cd), FadeIn(s), run_time=0.7); self.wait(0.3)
        self.say("recap: only one piece is new, turning a score into a probability", FG, 26)
        self.end()
        # 2 sigmoid from log-odds
        self.begin("sigmoid")
        self.play(FadeOut(VGroup(cards, srcs)), run_time=0.5)
        d = VGroup(MathTex(r"\ln\frac{p}{1-p}=z", font_size=40), MathTex(r"\frac{p}{1-p}=e^{z}", font_size=40),
                   MathTex(r"p=\frac{e^z}{1+e^z}=\frac{1}{1+e^{-z}}=\sigma(z)", font_size=40, color=C_SIGM)).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(LEFT * 3.4 + UP * 0.6)
        dl = Text("say: the log-odds is the score", font_size=22, color=MUTED).next_to(d, UP, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(dl), Write(d[0]), run_time=1.1); self.wait(0.4)
        self.play(Write(d[1]), run_time=0.9); self.wait(0.4)
        self.play(Write(d[2]), run_time=1.3)
        ax = Axes(x_range=[-5, 5, 1], y_range=[0, 1, 0.5], x_length=5.0, y_length=2.6,
                  axis_config={"color": MUTED, "stroke_width": 2}, tips=False).shift(RIGHT * 3.6 + UP * 0.5)
        cv = ax.plot(sgm, x_range=[-5, 5], color=C_SIGM, stroke_width=5)
        zl = Text("z", font_size=22, color=C_W).next_to(ax, RIGHT, buff=0.1)
        self.play(Create(ax), FadeIn(zl), Create(cv), run_time=1.2)
        vals = rows7(("z = 0 → odds 1:1 → p = ½", FG, 22), ("z = ln 3 ≈ 1.1 → odds 3:1 → p = ¾", FG, 22), ("z = −ln 3 → p = ¼ = 1 − ¾", FG, 22), buff=0.22).next_to(ax, DOWN, buff=0.45)
        for v in vals: self.play(FadeIn(v), run_time=0.55); self.wait(0.2)
        self.say("σ squashes any score into (0, 1); big z → near 1, very negative z → near 0", C_SIGM, 24)
        self.end()
        # 3 the loss
        self.begin("loss")
        self.play(FadeOut(VGroup(d, dl, ax, cv, zl, vals)), run_time=0.5)
        c1 = MathTex(r"y=1:\ L=-\ln p", font_size=42).shift(LEFT * 3.2 + UP * 1.4)
        c2 = MathTex(r"y=0:\ L=-\ln(1-p)", font_size=42).shift(RIGHT * 3.0 + UP * 1.4)
        self.play(Write(c1), run_time=1.0); self.play(Write(c2), run_time=1.0)
        both = MathTex(r"L=-\big[\,y\ln p+(1-y)\ln(1-p)\,\big]", font_size=46, color=C_LOSS).shift(UP * 0.15)
        bn = Text("the y's just switch the right term on (binary cross-entropy)", font_size=22, color=MUTED).next_to(both, DOWN, buff=0.2)
        self.play(Write(both), FadeIn(bn), run_time=1.4)
        ex = rows7(("p = 0.8, y = 1:  −ln 0.8 ≈ 0.22 nats", C_GOOD, 26), ("p = 0.8, y = 0:  −ln 0.2 ≈ 1.61 nats", C_BAD, 26), buff=0.3).shift(DOWN * 1.6)
        for it in ex: self.play(FadeIn(it), run_time=0.6); self.wait(0.3)
        self.say("the surprise of the true label, exactly as in Lesson 6", C_LOSS, 28)
        self.end()
        # 4 the sigmoid's slope
        self.begin("slope")
        self.play(FadeOut(VGroup(c1, c2, both, bn, ex)), run_time=0.5)
        s1 = MathTex(r"\frac{dp}{dz}=\frac{d}{dz}(1+e^{-z})^{-1}=\frac{e^{-z}}{(1+e^{-z})^2}", font_size=42).shift(UP * 1.3)
        s2 = MathTex(r"=\underbrace{\frac{1}{1+e^{-z}}}_{p}\cdot\underbrace{\frac{e^{-z}}{1+e^{-z}}}_{1-p}=p\,(1-p)", font_size=42, color=C_SIGM).shift(DOWN * 0.35)
        self.play(Write(s1), run_time=1.5); self.wait(0.4)
        self.play(Write(s2), run_time=1.6)
        n = Text("check: z = 0 → p = ½ → slope ¼, the steepest point of the S", font_size=24, color=MUTED).shift(DOWN * 2.0)
        self.play(FadeIn(n), run_time=0.7)
        self.say("the sigmoid's slope is p(1 − p): keep this, it cancels next", C_SIGM, 26)
        self.end()
        # 5 chain rule -> (p - y) x
        self.begin("chain")
        self.play(FadeOut(VGroup(s1, s2, n)), run_time=0.5)
        ch = MathTex(r"\frac{dL}{dw}=", r"\frac{dL}{dp}", r"\cdot", r"\frac{dp}{dz}", r"\cdot", r"\frac{dz}{dw}", font_size=46).next_to(title, DOWN, buff=0.4)
        ch[1].set_color(C_LOSS); ch[3].set_color(C_SIGM); ch[5].set_color(C_W)
        self.play(Write(ch), run_time=1.2)
        a = MathTex(r"y=1:\ \ -\frac1p\cdot p(1-p)\cdot x=-(1-p)\,x=(p-1)\,x", font_size=38).shift(UP * 0.55)
        b = MathTex(r"y=0:\ \ \frac{1}{1-p}\cdot p(1-p)\cdot x=p\,x=(p-0)\,x", font_size=38).shift(DOWN * 0.35)
        self.play(Write(a), run_time=1.5); self.wait(0.4)
        self.play(Write(b), run_time=1.5)
        r = MathTex(r"\frac{dL}{dw}=(p-y)\,x", font_size=54, color=C_ERR).shift(DOWN * 1.95)
        rb = SurroundingRectangle(r, color=C_ERR, buff=0.15)
        self.play(Write(r), Create(rb), run_time=1.0)
        self.say("how wrong the probability is, times the input", C_ERR, 28)
        self.end()
        # 6 worked example 1
        self.begin("ex1")
        self.play(FadeOut(VGroup(ch, a, b, r, rb)), run_time=0.5)
        h = Text("worked example 1 · x = 2, w = 0, y = 1, η = 0.5", font_size=28, color=C_W).next_to(title, DOWN, buff=0.45)
        self.play(FadeIn(h), run_time=0.6)
        ls = rows7(("z = w·x = 0   →   p = σ(0) = 0.5", FG, 28),
                   ("gradient = (p − y)·x = (0.5 − 1)·2 = −1", C_ERR, 28),
                   ("w ← 0 − 0.5·(−1) = 0.5", C_ETA, 28),
                   ("next: z = 0.5·2 = 1   →   p = σ(1) ≈ 0.73  ↑ toward y = 1", C_GOOD, 28), buff=0.4).shift(DOWN * 0.4)
        for it in ls: self.play(FadeIn(it, shift=RIGHT * 0.2), run_time=0.7); self.wait(0.4)
        self.say("p < y → negative gradient → w goes up → p goes up", FG, 28)
        self.end()
        # 7 worked example 2: edge cases
        self.begin("ex2")
        self.play(FadeOut(VGroup(h, ls)), run_time=0.5)
        h2 = Text("worked example 2 (edge cases) · x = 1", font_size=28, color=C_BAD).next_to(title, DOWN, buff=0.45)
        self.play(FadeIn(h2), run_time=0.6)
        tb = VGroup()
        data = [("confident & right", "p = 0.99, y = 1", "(0.99 − 1)·1 = −0.01", "almost nothing to learn", C_GOOD),
                ("confident & wrong", "p = 0.99, y = 0", "(0.99 − 0)·1 = 0.99", "big push (never more than |x|)", C_BAD),
                ("feature absent", "x = 0, any p", "(p − y)·0 = 0", "that weight doesn't move", MUTED)]
        for i, (k, v, g, m, c) in enumerate(data):
            rr = VGroup(Text(k, font_size=24, color=c), Text(v, font_size=24), Text(g, font_size=24, color=C_ERR), Text(m, font_size=20, color=MUTED))
            rr[0].move_to([-5.6, 0, 0], aligned_edge=LEFT); rr[1].move_to([-2.4, 0, 0], aligned_edge=LEFT)
            rr[2].move_to([0.7, 0, 0], aligned_edge=LEFT); rr[3].move_to([0.7, -0.38, 0], aligned_edge=LEFT)
            tb.add(rr.shift(DOWN * i * 1.05))
        tb.move_to(DOWN * 0.45)
        for rr in tb: self.play(FadeIn(rr), run_time=0.7); self.wait(0.4)
        self.say("the update is proportional to the error: effort goes where it's wrong", FG, 26)
        self.end()
        # 8 faded + same shape as MSE
        self.begin("faded")
        self.play(FadeOut(VGroup(h2, tb)), run_time=0.5)
        fd = Text("your turn: p = 0.2, y = 1, x = 3  →  gradient = ?", font_size=30, color=C_AL).shift(UP * 1.3)
        q = SurroundingRectangle(fd, color=C_AL, buff=0.15)
        self.play(FadeIn(fd), Create(q), run_time=0.9)
        sm = VGroup(Text("same shape as linear regression with ½·squared error:", font_size=24, color=MUTED),
                    MathTex(r"\tfrac{d}{dw}\,\tfrac12(\hat y-y)^2=(\hat y-y)\,x \qquad \tfrac{dL}{dw}=(p-y)\,x", font_size=48, color=C_ERR)).arrange(DOWN, buff=0.35).shift(DOWN * 0.6)
        self.play(FadeIn(sm[0]), run_time=0.6); self.play(Write(sm[1]), run_time=1.4)
        self.say("(prediction − truth) × input: the matched loss makes the gradient simple", C_ERR, 24)
        self.end(2.2)
