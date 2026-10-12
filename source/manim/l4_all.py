from manim import *
import numpy as np

config.background_color = "#0e1117"
FG = "#e6e6e6"; MUTED = "#8b93a7"; GRID = "#2a2f3a"
C_PI = "#b392f0"; C_OLD = "#e08fd0"; C_Y = "#7aa2f7"; C_X = "#5fc9b6"
C_R = "#9ece6a"; C_B = "#c3e88d"; C_RHO = "#ff9e64"; C_EPS = "#f7768e"
C_AL = "#e0af68"; C_ST = "#a9b1d6"; C_BAD = "#f7768e"; C_GOOD = "#9ece6a"
Text.set_default(color=FG, font="DejaVu Sans")
MathTex.set_default(color=FG)

def softmax(z):
    z = np.array(z, float); e = np.exp(z - z.max()); return e / e.sum()

RESP = ["Paris is the capital.", "Paris.", "Maybe Lyon?", "As an AI, I can't…", "Lyon."]
REW = [1.0, 0.8, 0.4, 0.2, 0.0]
INIT = [0, 0, 0, 0.4, 0.6]
BARW = 6.0

def caption(s, size=30, color=FG):
    return Text(s, font_size=size, color=color)

class Bars(VGroup):
    """Horizontal probability bars for the 5 toy responses."""
    def __init__(self, p, hl=None, show_r=True, width=BARW, label_w=3.6, old=None, ratios=None, eps=0.2, fs=28, labels=None):
        super().__init__()
        for i, (t, r) in enumerate(zip(labels or RESP, REW)):
            y = -i * 0.66
            lab = Text(t, font_size=fs, color=C_Y).move_to([0, y, 0], aligned_edge=RIGHT).shift(LEFT * 0.2)
            track = Rectangle(width=width, height=0.44, stroke_width=0, fill_color=GRID, fill_opacity=1).move_to([width / 2, y, 0])
            w = max(0.02, p[i] * width)
            bar = Rectangle(width=w, height=0.44, stroke_width=0, fill_color=C_PI, fill_opacity=0.9).move_to([w / 2, y, 0])
            if hl == i:
                bar.set_stroke(C_AL, 3)
            pct = Text(f"{p[i]*100:.0f}%", font_size=fs, color=FG).move_to([width + 0.2, y, 0], aligned_edge=LEFT)
            g = VGroup(lab, track, bar, pct)
            if old is not None:
                g.add(Line([old[i] * width, y - 0.26, 0], [old[i] * width, y + 0.26, 0], color=C_OLD, stroke_width=4))
            if ratios is not None:
                rr = ratios[i]; bad = rr > 1 + eps + 1e-9 or rr < 1 - eps - 1e-9
                g.add(Text(f"ρ {rr:.2f}", font_size=fs, color=C_BAD if bad else C_GOOD).move_to([width + 1.25, y, 0], aligned_edge=LEFT))
            elif show_r:
                g.add(Text(f"R={r:.1f}", font_size=fs, color=C_R).move_to([width + 1.3, y, 0], aligned_edge=LEFT))
            self.add(g)


def morph(old, new):
    """Animations morphing row-groups old->new; changed Text cross-fades instead of glyph-morphing."""
    anims, post = [], []
    for go, gn in zip(old, new):
        for a, b in zip(list(go.submobjects), list(gn.submobjects)):
            if isinstance(a, Text) and a.text != b.text:
                b.set_fill(opacity=0); go.add(b)
                anims += [a.animate.set_fill(opacity=0), b.animate.set_fill(opacity=1)]
                post.append((go, a, b))
            elif isinstance(a, Text):
                continue
            else:
                anims.append(Transform(a, b))
    return anims, post

def finish(post):
    for go, a, b in post:
        go.remove(b); idx = go.submobjects.index(a); go.submobjects[idx] = b

def play_morph(scene, old, new, **kw):
    anims, post = morph(old, new); scene.play(*anims, **kw); finish(post)

def swap(scene, old, new, **kw):
    scene.play(FadeTransform(old, new), **kw); return new

def reinforce(th, y, coef):
    p = softmax(th); g = -p.copy(); g[y] += 1
    return np.array(th, float) + coef * g


import json, os
STEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps")
HOLD = 1.8
BOT = dict(edge=DOWN, buff=0.3)

class StepScene(Scene):
    """Records [start, end] of every step from the renderer clock; every step ends with a hold."""
    def setup(self):
        self.steps = []; self._open = None; self.cap = None
    def begin(self, key):
        self._open = (key, self.renderer.time)
    def end(self, hold=HOLD):
        self.wait(hold)
        k, t0 = self._open
        self.steps.append({"key": k, "start": round(t0, 3), "end": round(self.renderer.time, 3), "hold": hold})
    def say(self, txt, color=FG, size=32, rt=0.8):
        """Replace the bottom caption (fade old out, new in)."""
        new = caption(txt, size, color).to_edge(DOWN, buff=0.3)
        anims = [FadeIn(new, shift=UP * 0.1)]
        if self.cap is not None: anims.append(FadeOut(self.cap))
        self.play(*anims, run_time=rt); self.cap = new
    def unsay(self, rt=0.5):
        if self.cap is not None: self.play(FadeOut(self.cap), run_time=rt); self.cap = None
    def tear_down(self):
        os.makedirs(STEP_DIR, exist_ok=True)
        with open(os.path.join(STEP_DIR, type(self).__name__ + ".json"), "w") as f:
            json.dump({"duration": round(self.renderer.time, 3), "steps": self.steps}, f, indent=1)

# ---------------------------------------------------------------- Chapter 1
class Reinforce(StepScene):
    def construct(self):
        title = caption("REINFORCE: push up whatever you sampled", 36).to_edge(UP, buff=0.4)
        f = MathTex(r"\theta", r"\leftarrow", r"\theta", r"+", r"\alpha", r"\cdot", r"R", r"\cdot",
                    r"\nabla_\theta \log", r"\pi_\theta", r"(", r"y", r"\mid", r"x", r")", font_size=60)
        cols = {0: C_PI, 2: C_PI, 4: C_AL, 6: C_R, 9: C_PI, 11: C_Y, 13: C_X}
        for k, c in cols.items(): f[k].set_color(c)
        f.next_to(title, DOWN, buff=0.5)
        # step 1: the update rule
        self.begin("rule")
        self.play(FadeIn(title), run_time=0.8)
        self.play(Write(f[0:4]), run_time=1.2)
        l1 = Text("step size", font_size=28, color=C_AL).next_to(f[4], DOWN, buff=0.25)
        self.play(Write(f[4]), FadeIn(l1), run_time=1.2); self.wait(0.6)
        self.play(Write(f[5]), run_time=0.3)
        l2 = Text("reward", font_size=28, color=C_R).next_to(f[6], DOWN, buff=0.85)
        self.play(Write(f[6]), FadeIn(l2), run_time=1.2); self.wait(0.6)
        self.play(Write(f[7]), run_time=0.3)
        l3 = Text("make this response more likely", font_size=28, color=C_PI).next_to(f[8:15], DOWN, buff=1.45)
        self.play(Write(f[8:15]), FadeIn(l3), run_time=1.6)
        self.say("bigger reward → bigger push on the sampled response y")
        self.end()
        # step 2: the toy
        self.begin("toy")
        self.play(FadeOut(VGroup(l1, l2, l3)), run_time=0.6)
        th = np.array(INIT, float)
        bars = Bars(softmax(th)).scale(0.95).move_to(DOWN * 1.15 + RIGHT * 0.2)
        prompt = Text('Prompt x: "What is the capital of France?"', font_size=30, color=C_X).next_to(bars, UP, buff=0.3)
        self.play(FadeIn(prompt), FadeIn(bars), run_time=1.2)
        self.say("5 possible answers · bar = probability · green = reward")
        self.end()

        def update(th, y, alpha, R, b=0.0):
            th2 = reinforce(th, y, alpha * (R - b))
            play_morph(self, bars, Bars(softmax(th), hl=y).scale(0.95).move_to(bars), run_time=0.7)
            self.wait(0.9)
            play_morph(self, bars, Bars(softmax(th2), hl=y).scale(0.95).move_to(bars), run_time=1.6)
            self.wait(0.6)
            return th2
        # step 3: small steps
        self.begin("small")
        self.say("α = 1: sample “Paris.” (R = 0.8), then “Paris is the capital.” (R = 1.0)")
        th = update(th, 1, 1.0, 0.8)
        th = update(th, 0, 1.0, 1.0)
        self.say("small steps: good answers slowly gain", C_GOOD)
        self.end()
        # step 4: big step + unlucky sample
        self.begin("collapse")
        th = np.array(INIT, float)
        self.say("Reset. Now a big step α = 8 and an unlucky sample", C_AL)
        play_morph(self, bars, Bars(softmax(th)).scale(0.95).move_to(bars), run_time=1.0)
        self.wait(0.6)
        self.say("“As an AI…”: R = 0.2 is still > 0 → pushed up hard", C_BAD)
        th = update(th, 3, 8.0, 0.2)
        self.say("now it's likely → sampled again → pushed up again", C_BAD)
        th = update(th, 3, 8.0, 0.2)
        self.say("collapse onto a weak answer", C_BAD, 36)
        self.end()
        # step 5: baseline
        self.begin("baseline")
        f2 = MathTex(r"\theta", r"\leftarrow", r"\theta", r"+", r"\alpha", r"\cdot", r"(R-b)", r"\cdot",
                     r"\nabla_\theta \log", r"\pi_\theta", r"(", r"y", r"\mid", r"x", r")", font_size=60).move_to(f)
        for k, c in cols.items(): f2[k].set_color(c)
        f2[6].set_color(C_R)
        th = np.array(INIT, float)
        a1, p1 = morph(bars, Bars(softmax(th)).scale(0.95).move_to(bars))
        self.play(TransformMatchingTex(f, f2), *a1, run_time=1.5); finish(p1)
        self.say("Fix: subtract a baseline b = typical reward ≈ 0.40", C_R)
        self.wait(0.8)
        b = float(softmax(th) @ np.array(REW))
        th = update(th, 3, 8.0, 0.2, b=b)
        self.say(f"same sample: A = 0.2 − {b:.2f} = {0.2-b:+.2f} → pushed DOWN".replace("-0", "−0"), C_GOOD)
        self.end(2.2)

# ---------------------------------------------------------------- Chapter 2
class Clip(StepScene):
    def construct(self):
        # step 1: ratio
        self.begin("ratio")
        r = MathTex(r"\rho", r"=", r"{\pi_\theta(y\mid x)", r"\over", r"\pi_{\text{old}}(y\mid x)}", font_size=64)
        r[0].set_color(C_RHO); r[2].set_color(C_PI); r[4].set_color(C_OLD)
        self.play(Write(r[0:2]), run_time=1)
        n1 = Text("current policy", font_size=26, color=C_PI).next_to(r, RIGHT, buff=0.5).shift(UP * 0.45)
        self.play(Write(r[2]), FadeIn(n1), run_time=1.3); self.wait(0.5)
        n2 = Text("policy that sampled it", font_size=26, color=C_OLD).next_to(r, RIGHT, buff=0.5).shift(DOWN * 0.45)
        self.play(Write(r[3:]), FadeIn(n2), run_time=1.3); self.wait(0.5)
        ex = Text("ρ = 1.5  →  now 50% more likely than when it was sampled", font_size=30, color=C_RHO).next_to(r, DOWN, buff=0.7)
        self.play(FadeIn(ex), run_time=0.8)
        self.end()
        # step 2: objective
        self.begin("objective")
        self.play(FadeOut(VGroup(r, n1, n2, ex)), run_time=0.7)
        L = MathTex(r"L", r"=", r"\min\big(", r"\rho", r"A", r",\;", r"\mathrm{clip}(", r"\rho", r",\,1-", r"\epsilon", r",\,1+", r"\epsilon", r")", r"A", r"\big)", font_size=56)
        for k, c in {3: C_RHO, 4: C_R, 7: C_RHO, 9: C_EPS, 11: C_EPS, 13: C_R}.items(): L[k].set_color(c)
        self.play(Write(L[0:5]), run_time=1.3)
        t1 = Text("the plain objective ρ·A", font_size=26, color=MUTED).next_to(L[3:5], DOWN, buff=0.5)
        self.play(FadeIn(t1), run_time=0.6); self.wait(0.8)
        self.play(Write(L[5:]), run_time=1.6)
        t2 = Text("same, but ρ clamped to [1−ε, 1+ε]", font_size=26, color=MUTED).next_to(L[6:14], DOWN, buff=1.1)
        self.play(FadeIn(t2), run_time=0.6)
        self.say("PPO takes the smaller (pessimistic) of the two")
        self.end()
        # step 3: A > 0 unclipped
        self.begin("apos")
        self.unsay()
        self.play(FadeOut(t1), FadeOut(t2), L.animate.scale(0.85).to_edge(UP, buff=0.35), run_time=1)
        ax = Axes(x_range=[0, 2, 0.5], y_range=[-1.6, 1.6, 0.5], x_length=9.5, y_length=4.4,
                  axis_config={"color": MUTED, "include_tip": False}, x_axis_config={"numbers_to_include": [0.5, 1, 1.5, 2], "font_size": 28}).shift(DOWN * 0.45 + LEFT * 0.6)
        xl = MathTex(r"\rho", color=C_RHO, font_size=44).next_to(ax.x_axis, RIGHT, buff=0.2)
        self.play(Create(ax), FadeIn(xl), run_time=1.2)
        eps = 0.2
        tagA = MathTex(r"A=+1", color=C_R, font_size=48).to_corner(UR, buff=0.5).shift(DOWN * 1.3)
        self.play(FadeIn(tagA), run_time=0.6)
        unc = DashedVMobject(ax.plot(lambda x: x, x_range=[0, 2], color=MUTED), num_dashes=40)
        ul = Text("ρ·A (no clip)", font_size=26, color=MUTED).next_to(ax.c2p(1.25, 1.55), LEFT, buff=0.1)
        self.play(Create(unc), FadeIn(ul), run_time=1.5)
        self.say("good response (A > 0): ρ·A rewards pushing ρ up forever")
        self.end()
        # step 4: clip
        self.begin("clip")
        vlines = VGroup(*[DashedLine(ax.c2p(v, -1.6), ax.c2p(v, 1.6), color=C_EPS, stroke_width=2) for v in (1 - eps, 1 + eps)])
        vl = VGroup(MathTex(r"1-\epsilon", color=C_EPS, font_size=32).next_to(vlines[0], UP, buff=0.08),
                    MathTex(r"1+\epsilon", color=C_EPS, font_size=32).next_to(vlines[1], UP, buff=0.08))
        self.play(Create(vlines), FadeIn(vl), run_time=1)
        def Lc(x, A): return min(x * A, np.clip(x, 1 - eps, 1 + eps) * A)
        cur = ax.plot(lambda x: Lc(x, 1), x_range=[0, 2, 0.005], color=C_R, stroke_width=6)
        self.play(Create(cur), run_time=2)
        zone = Rectangle(width=ax.c2p(2, 0)[0] - ax.c2p(1 + eps, 0)[0], height=ax.y_length, stroke_width=0, fill_color=C_EPS, fill_opacity=0.15)
        zone.move_to(ax.c2p((2 + 1 + eps) / 2, 0))
        zt = Text("flat → gradient = 0", font_size=28, color=C_EPS).move_to(ax.c2p(1.6, -0.6))
        self.play(FadeIn(zone), FadeIn(zt), run_time=1)
        self.say("past 1 + ε (ε = 0.2) the objective is flat → no more push", C_EPS)
        self.end()
        # step 5: worked example
        self.begin("example")
        t = ValueTracker(1.0); sgn = ValueTracker(1)
        dot = always_redraw(lambda: Dot(ax.c2p(t.get_value(), Lc(t.get_value(), 1 if sgn.get_value() > 0 else -1)), color=C_AL, radius=0.12))
        self.unsay()
        self.add(dot)
        self.play(t.animate.set_value(1.5), run_time=2)
        w = MathTex(r"\rho=1.5,\ A=+1:\ \min(1.5,\ 1.2)=1.2", r"\Rightarrow\ \nabla=0", font_size=40).to_edge(DOWN, buff=0.25)
        w[1].set_color(C_EPS)
        self.play(Write(w), run_time=1.5); self.cap = w
        self.end(2.2)
        # step 6: A < 0
        self.begin("aneg")
        self.unsay()
        tagB = MathTex(r"A=-1", color=C_BAD, font_size=48).move_to(tagA)
        unc2 = DashedVMobject(ax.plot(lambda x: -x, x_range=[0, 1.6], color=MUTED), num_dashes=36)
        cur2 = ax.plot(lambda x: Lc(x, -1), x_range=[0, 1.6, 0.005], color=C_BAD, stroke_width=6)
        zone2 = Rectangle(width=ax.c2p(1 - eps, 0)[0] - ax.c2p(0, 0)[0], height=ax.y_length, stroke_width=0, fill_color=C_EPS, fill_opacity=0.15).move_to(ax.c2p((1 - eps) / 2, 0))
        zt2 = Text("flat → gradient = 0", font_size=28, color=C_EPS).move_to(ax.c2p(0.4, 0.6))
        self.play(Transform(tagA, tagB), Transform(unc, unc2), Transform(cur, cur2), Transform(zone, zone2), Transform(zt, zt2), FadeOut(ul), sgn.animate.set_value(-1), run_time=2)
        self.wait(1)
        w2 = MathTex(r"\rho=1.5,\ A=-1:\ \min(-1.5,\ -1.2)=-1.5", r"\Rightarrow\ \text{still active}", font_size=40).to_edge(DOWN, buff=0.25)
        w2[1].set_color(C_GOOD)
        self.play(Write(w2), run_time=1.5); self.cap = w2
        self.end(2.2)

# ---------------------------------------------------------------- Chapter 3
SHORT = ["Paris (full)", "Paris.", "Lyon?", "As an AI…", "Lyon."]
BATCH_Y = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
NOISE = [0.05, -0.1, 0.12, -0.05, 0.08, -0.12, 0.1, 0.0, 0.07, -0.08, 0.15, -0.02, 0.1, -0.06, 0.03, -0.1]

def run_epochs(K=10, lr=3.0, eps=0.2):
    pold = softmax(INIT)
    R = np.array([REW[y] + n for y, n in zip(BATCH_Y, NOISE)]); A = R - R.mean()
    out = {"naive": [np.array(INIT, float)], "ppo": [np.array(INIT, float)]}
    for _ in range(K):
        for kind in out:
            th = out[kind][-1]; p = softmax(th); g = np.zeros(5)
            for y, a in zip(BATCH_Y, A):
                rho = p[y] / pold[y]; coef = a
                if kind == "ppo":
                    coef = 0 if ((a > 0 and rho > 1 + eps) or (a < 0 and rho < 1 - eps)) else a * rho
                e = -p.copy(); e[y] += 1; g += coef * e / len(BATCH_Y)
            out[kind].append(th + lr * g)
    return pold, A, out

class NaiveVsPPO(StepScene):
    def construct(self):
        pold, A, out = run_epochs()
        def panel(th):
            p = softmax(th)
            return Bars(p, old=pold, ratios=p / pold, width=2.7, fs=28, labels=SHORT).scale(0.85)
        # step 1: setup
        self.begin("setup")
        title = caption("Same batch, K = 10 gradient steps", 36).to_edge(UP, buff=0.4)
        sub = caption("pink tick = π_old   ·   ρ = π_θ / π_old", 26, MUTED).next_to(title, DOWN, buff=0.15)
        N = panel(out["naive"][0]).move_to(LEFT * 3.5 + DOWN * 0.55)
        P = panel(out["ppo"][0]).move_to(RIGHT * 3.5 + DOWN * 0.55)
        hN = caption("Naive: maximize ρ·A", 32); hP = caption("PPO: min(ρA, clip(ρ)A)", 32)
        hN.next_to(N, UP, buff=0.35); hP.next_to(P, UP, buff=0.35)
        self.play(FadeIn(title), FadeIn(sub), run_time=0.8)
        self.play(FadeIn(N), FadeIn(hN), run_time=0.8); self.play(FadeIn(P), FadeIn(hP), run_time=0.8)
        self.say("16 responses sampled ONCE from π_old · A = R − batch mean", C_R, 30)
        self.end(2.2)
        ep = None
        def epochs(k0, k1, rt):
            nonlocal N, P, ep
            for k in range(k0, k1 + 1):
                a1, p1 = morph(N, panel(out["naive"][k]).move_to(N)); a2, p2 = morph(P, panel(out["ppo"][k]).move_to(P))
                e2 = Text(f"epoch {k}", font_size=32, color=C_AL).to_edge(DOWN, buff=1.1)
                anims = [*a1, *a2, (FadeTransform(ep, e2) if ep else FadeIn(e2))]
                self.play(*anims, run_time=rt); finish(p1); finish(p2); ep = e2
                self.wait(0.25)
        # step 2: first epochs
        self.begin("early")
        self.say("epochs 1–3: both start moving the same way", FG, 30)
        epochs(1, 3, 1.3)
        self.end()
        # step 3: later epochs
        self.begin("late")
        self.say("epochs 4–10: naive keeps pushing; clipped PPO samples stop", FG, 30)
        epochs(4, 10, 1.0)
        self.end()
        # step 4: verdict
        self.begin("verdict")
        self.unsay()
        mn = (softmax(out["naive"][-1]) / pold).max(); mp = (softmax(out["ppo"][-1]) / pold).max()
        vN = caption(f"max ρ = {mn:.1f}×: runs away", 32, C_BAD).next_to(N, DOWN, buff=0.45)
        vP = caption(f"max ρ = {mp:.2f}×: bounded", 32, C_GOOD).next_to(P, DOWN, buff=0.45)
        self.play(FadeOut(ep), FadeIn(vN), FadeIn(vP), run_time=1)
        self.end(2.5)

# ---------------------------------------------------------------- Chapter 4
class GRPO(StepScene):
    def construct(self):
        f = MathTex(r"A_i", r"=", r"{r_i", r"-", r"\mu", r"\over", r"\sigma}", font_size=64)
        f[0].set_color(C_R); f[2].set_color(C_R); f[4].set_color(C_ST); f[6].set_color(C_ST)
        f.to_edge(UP, buff=0.5)
        rew = [1, 0, 0, 1, 1, 0, 0, 0]
        ax = Axes(x_range=[0, 8, 1], y_range=[-1.6, 1.6, 0.5], x_length=12, y_length=4.6).shift(DOWN * 0.55)
        zero = Line(ax.c2p(0, 0), ax.c2p(8, 0), color=MUTED)
        def bars(vals, color_by_sign=False):
            g = VGroup()
            for i, v in enumerate(vals):
                c = C_R if not color_by_sign else (C_GOOD if v > 1e-9 else C_BAD if v < -1e-9 else MUTED)
                h = max(0.03, abs(ax.c2p(0, v)[1] - ax.c2p(0, 0)[1]))
                rct = Rectangle(width=0.8, height=h, stroke_width=0, fill_color=c, fill_opacity=0.9)
                rct.move_to(ax.c2p(i + 0.5, 0), aligned_edge=DOWN if v >= 0 else UP)
                lab = Text(f"{v:.2f}".replace("-", "−"), font_size=28, color=FG).next_to(rct, UP if v >= 0 else DOWN, buff=0.08)
                g.add(VGroup(rct, lab))
            return g
        names = VGroup(*[MathTex(f"y_{i+1}", font_size=36, color=C_Y).move_to(ax.c2p(i + 0.5, -1.55)) for i in range(8)])
        # step 1
        self.begin("rewards")
        B = bars(rew)
        self.play(Write(f[0:3]), run_time=1.2)
        self.play(Create(zero), FadeIn(B), FadeIn(names), run_time=1.2)
        self.say("8 responses to one prompt, reward = 1 if correct, else 0")
        self.end()
        # step 2
        self.begin("mean")
        m = float(np.mean(rew)); s = float(np.std(rew, ddof=1))
        ml = DashedLine(ax.c2p(0, m), ax.c2p(8, m), color=C_ST)
        mt = MathTex(rf"\mu={m:.3f}", color=C_ST, font_size=38).next_to(ml.get_left(), UP + RIGHT, buff=0.1)
        self.play(Write(f[3:5]), run_time=1.2)
        self.play(Create(ml), FadeIn(mt), run_time=1.2)
        self.say("group mean μ = 3/8 is the baseline: no critic needed")
        self.end()
        # step 3
        self.begin("subtract")
        c = [v - m for v in rew]
        a1, p1 = morph(B, bars(c, True))
        self.play(*a1, ml.animate.move_to(ax.c2p(4, 0)), FadeOut(mt), run_time=2); finish(p1)
        self.say("subtract μ: above average → positive, below → negative")
        self.end()
        # step 4
        self.begin("divide")
        a = [v / s for v in c]
        self.play(Write(f[5:]), run_time=1.2)
        a1, p1 = morph(B, bars(a, True))
        self.play(*a1, run_time=2); finish(p1)
        self.say(f"divide by σ = {s:.2f}: these are the advantages A_i")
        self.end(2.2)
        # step 5
        self.begin("equal")
        self.say("Now suppose all 8 are correct…")
        a1, p1 = morph(B, bars([1] * 8)); self.play(FadeOut(ml), *a1, run_time=1.5); finish(p1)
        self.wait(1)
        a1, p1 = morph(B, bars([0] * 8, True)); self.play(*a1, run_time=1.8); finish(p1)
        self.say("σ = 0 → every A = 0 → no learning signal from this prompt", C_BAD)
        self.end(2.5)
# Lesson 4 · ML fundamentals for pen-and-paper interviews (Guava MLE)

C_BV = "#7aa2f7"; C_OV = "#f7768e"; C_UN = "#e0af68"; C_RG = "#9ece6a"
C_L2 = "#b392f0"; C_L1 = "#7dcfff"; C_DO = "#ff9e64"; C_LOSS = "#e08fd0"
C_GD = "#c3e88d"; C_SIG = "#7aa2f7"; C_BP = "#f7768e"

# ---------------------------------------------------------------- Ch1: Metrics recall
class MetricsRecall(StepScene):
    def construct(self):
        title = caption("Lesson 3 weak spots · retrieval before new material", 32).to_edge(UP, buff=0.35)
        # 1 AUROC pairs
        self.begin("auroc")
        self.play(FadeIn(title), run_time=0.7)
        h = Text("AUROC = P(random + scores above random −)", font_size=30, color=C_GOOD).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(h), run_time=0.7)
        pos = Text("positives  {0.9, 0.5}", font_size=28, color=C_GOOD).shift(LEFT * 3.2 + UP * 0.6)
        neg = Text("negatives  {0.7, 0.5, 0.2}", font_size=28, color=C_BAD).shift(RIGHT * 2.8 + UP * 0.6)
        self.play(FadeIn(pos), FadeIn(neg), run_time=0.8)
        rows = VGroup(
            Text("(0.9, 0.7) → 1    (0.9, 0.5) → 1    (0.9, 0.2) → 1", font_size=26),
            Text("(0.5, 0.7) → 0    (0.5, 0.5) → ½    (0.5, 0.2) → 1", font_size=26),
            Text("sum = 4.5   ·   6 pairs   ·   AUROC = 4.5/6 = 0.75", font_size=28, color=C_AL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(DOWN * 1.0)
        self.play(FadeIn(rows), run_time=1.2)
        self.say("ties count as ½ · list every (pos, neg) pair", C_AL, 28)
        self.end()
        # 2 Fβ
        self.begin("fbeta")
        self.play(FadeOut(VGroup(h, pos, neg, rows)), run_time=0.5)
        f = MathTex(r"F_\beta=\frac{(1+\beta^2)PR}{\beta^2 P+R}", font_size=52).shift(UP * 1.4)
        self.play(Write(f), run_time=1.4)
        why = VGroup(
            Text("weights on 1/P and 1/R:  1/(1+β²)  and  β²/(1+β²)", font_size=26, color=MUTED),
            Text("they sum to 1 → if P = R = x then Fβ = x", font_size=26, color=MUTED),
            Text("β² (not β): equal ∂F/∂P and ∂F/∂R when R/P = β", font_size=26, color=C_AL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(DOWN * 0.5)
        self.play(FadeIn(why), run_time=1.2)
        ex = Text("P=0.8, R=0.4 → F1≈0.53 · F2≈0.44 · F0.5≈0.67", font_size=28, color=C_GOOD).to_edge(DOWN, buff=1.1)
        self.play(FadeIn(ex), run_time=0.7)
        self.say("β>1 favors recall (escalation); β<1 favors precision", FG, 28)
        self.end()
        # 3 PR vs ROC
        self.begin("pr")
        self.play(FadeOut(VGroup(f, why, ex)), run_time=0.5)
        t = Text("Rare positives (1% fraud): AUROC can look great while precision is low", font_size=28).shift(UP * 2.0)
        self.play(FadeIn(t), run_time=0.7)
        bullets = VGroup(
            Text("ROC random baseline = 0.5  (any prevalence)", font_size=28, color=C_BV),
            Text("PR  random baseline = prevalence  (0.01 if 1% positive)", font_size=28, color=C_DO),
            Text("At TPR 0.9, FPR 0.05 on 10k (100 +): TP=90, FP≈495 → P≈0.15", font_size=26, color=C_BAD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(DOWN * 0.2)
        self.play(FadeIn(bullets), run_time=1.2)
        self.say("use PR-AUC when positives are rare (fraud, escalations)", C_DO, 28)
        self.end()
        # 4 WER
        self.begin("wer")
        self.play(FadeOut(VGroup(t, bullets)), run_time=0.5)
        w = MathTex(r"\mathrm{WER}=\frac{S+D+I}{N_{\mathrm{ref}}}", font_size=56).shift(UP * 1.5)
        self.play(Write(w), run_time=1.2)
        e1 = Text("ref “yes” (N=1) · ASR “yes yes okay” → S=0 D=0 I=2 → WER=200%", font_size=26, color=C_BAD).shift(UP * 0.2)
        e2 = Text("WER = word-level Levenshtein / reference words · can exceed 100%", font_size=26, color=MUTED).shift(DOWN * 0.6)
        self.play(FadeIn(e1), FadeIn(e2), run_time=1.0)
        self.say("insertions count; denominator is only the reference length", FG, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch2: Bias–variance
class BiasVariance(StepScene):
    def construct(self):
        title = caption("Train vs validation error · diagnose underfit / overfit", 32).to_edge(UP, buff=0.35)
        # 1 setup
        self.begin("setup")
        self.play(FadeIn(title), run_time=0.7)
        axes = Axes(x_range=[0, 10, 2], y_range=[0, 1.1, 0.25], x_length=9, y_length=4.2,
                    axis_config={"color": MUTED, "stroke_width": 2}, tips=False).shift(DOWN * 0.35)
        xl = Text("model complexity →", font_size=24, color=MUTED).next_to(axes, DOWN, buff=0.15)
        yl = Text("error", font_size=24, color=MUTED).next_to(axes, LEFT, buff=0.1)
        self.play(Create(axes), FadeIn(xl), FadeIn(yl), run_time=1.2)
        self.say("x-axis: richer models (more params / less regularization)", FG, 28)
        self.end()
        # 2 curves
        self.begin("curves")
        train = axes.plot(lambda x: 0.55 * np.exp(-0.35 * x) + 0.08, color=C_GOOD, stroke_width=5)
        val = axes.plot(lambda x: 0.25 + 0.35 * ((x - 4.5) / 4.5) ** 2 + 0.05 * np.exp(-0.4 * x), color=C_BAD, stroke_width=5)
        lt = Text("train error", font_size=24, color=C_GOOD).move_to(axes.c2p(8.2, 0.18))
        lv = Text("val error", font_size=24, color=C_BAD).move_to(axes.c2p(8.2, 0.72))
        self.play(Create(train), FadeIn(lt), run_time=1.2)
        self.play(Create(val), FadeIn(lv), run_time=1.2)
        self.say("train falls; val falls then rises — the rise is overfitting", FG, 28)
        self.end()
        # 3 underfit
        self.begin("under")
        u = DashedLine(axes.c2p(1.2, 0), axes.c2p(1.2, 1.05), color=C_UN, stroke_width=3)
        ul = Text("underfit", font_size=26, color=C_UN).next_to(u, UP, buff=0.1)
        self.play(Create(u), FadeIn(ul), run_time=0.9)
        self.say("both errors high · model too simple · add capacity or train longer", C_UN, 28)
        self.end()
        # 4 overfit
        self.begin("over")
        o = DashedLine(axes.c2p(8.0, 0), axes.c2p(8.0, 1.05), color=C_OV, stroke_width=3)
        ol = Text("overfit", font_size=26, color=C_OV).next_to(o, UP, buff=0.1)
        self.play(Create(o), FadeIn(ol), run_time=0.9)
        gap = Line(axes.c2p(8.25, 0.12), axes.c2p(8.25, 0.68), color=C_OV, stroke_width=4)
        gl = Text("gap", font_size=24, color=C_OV).next_to(gap, RIGHT, buff=0.12)
        self.play(Create(gap), FadeIn(gl), run_time=0.8)
        self.say("train low, val high · memorizing · regularize / more data / early stop", C_OV, 26)
        self.end()
        # 5 sweet
        self.begin("sweet")
        s = DashedLine(axes.c2p(4.5, 0), axes.c2p(4.5, 1.05), color=C_RG, stroke_width=3)
        sl = Text("sweet spot", font_size=26, color=C_RG).next_to(s, UP, buff=0.1)
        self.play(Create(s), FadeIn(sl), run_time=0.9)
        self.say("pick complexity / early-stop where validation error is lowest", C_RG, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch3: Regularization
class Regularize(StepScene):
    def construct(self):
        title = caption("L2 · L1 · dropout — stop the model from memorizing", 32).to_edge(UP, buff=0.35)
        # 1 L2
        self.begin("l2")
        self.play(FadeIn(title), run_time=0.7)
        f = MathTex(r"L_{\mathrm{L2}}=L_{\mathrm{data}}+\lambda\sum_j w_j^2", font_size=48).shift(UP * 1.6)
        self.play(Write(f), run_time=1.3)
        pts = VGroup(
            Text("also called weight decay: each step multiplies w by (1 − 2ηλ)", font_size=26, color=MUTED),
            Text("shrinks weights toward 0 · prefers many small weights", font_size=26, color=C_L2),
            Text("prior view: Gaussian prior on w  →  −log prior ∝ ‖w‖²", font_size=26, color=C_L2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(DOWN * 0.4)
        self.play(FadeIn(pts), run_time=1.2)
        self.say("λ bigger → stronger shrink · less able to fit noise", C_L2, 28)
        self.end()
        # 2 L1
        self.begin("l1")
        self.play(FadeOut(VGroup(f, pts)), run_time=0.5)
        f2 = MathTex(r"L_{\mathrm{L1}}=L_{\mathrm{data}}+\lambda\sum_j |w_j|", font_size=48).shift(UP * 1.8)
        self.play(Write(f2), run_time=1.2)
        # diamond vs circle constraint sketch
        circ = Circle(1.1, color=C_L2, stroke_width=3).shift(LEFT * 3.2 + DOWN * 0.3)
        diam = Polygon(*[np.array(p) for p in [[0,1.2,0],[1.2,0,0],[0,-1.2,0],[-1.2,0,0]]], color=C_L1, stroke_width=3).shift(RIGHT * 3.0 + DOWN * 0.3)
        lc = Text("L2 ball", font_size=24, color=C_L2).next_to(circ, DOWN, buff=0.2)
        ld = Text("L1 diamond", font_size=24, color=C_L1).next_to(diam, DOWN, buff=0.2)
        tip = Text("loss contours hit a corner → some wi = 0 (sparse)", font_size=26, color=C_L1).shift(DOWN * 2.4)
        self.play(Create(circ), Create(diam), FadeIn(lc), FadeIn(ld), run_time=1.4)
        self.play(FadeIn(tip), run_time=0.7)
        self.say("L1 → Laplace prior · corners encourage exact zeros", C_L1, 28)
        self.end()
        # 3 dropout
        self.begin("drop")
        self.play(FadeOut(VGroup(f2, circ, diam, lc, ld, tip)), run_time=0.5)
        nodes = VGroup(*[Circle(0.28, color=C_DO, fill_opacity=0.25, stroke_width=3) for _ in range(6)]).arrange(RIGHT, buff=0.55).shift(UP * 0.6)
        labs = VGroup(*[Text(str(i + 1), font_size=22).move_to(nodes[i]) for i in range(6)])
        self.play(FadeIn(nodes), FadeIn(labs), run_time=0.9)
        self.play(*[nodes[i].animate.set_fill(C_BAD, 0.5).set_stroke(C_BAD) for i in (1, 4)],
                  *[labs[i].animate.set_color(C_BAD) for i in (1, 4)], run_time=1.0)
        note = VGroup(
            Text("each training step: randomly zero units (p ≈ 0.5 hidden)", font_size=26, color=MUTED),
            Text("forces redundancy · at test time scale by keep-prob (or inverted dropout)", font_size=26, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(DOWN * 1.3)
        self.play(FadeIn(note), run_time=1.0)
        self.say("dropout ≈ training a bag of thinned networks", C_DO, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch4: Splits & leakage
class SplitsLeak(StepScene):
    def construct(self):
        title = caption("Train / val / test · k-fold · data leakage (speech)", 32).to_edge(UP, buff=0.35)
        # 1 three-way
        self.begin("three")
        self.play(FadeIn(title), run_time=0.7)
        boxes = VGroup(
            VGroup(Rectangle(width=2.8, height=1.4, stroke_color=C_GOOD, fill_color=C_GOOD, fill_opacity=0.15, stroke_width=3), Text("train\nfit weights", font_size=26, color=C_GOOD, line_spacing=1.1)),
            VGroup(Rectangle(width=2.8, height=1.4, stroke_color=C_AL, fill_color=C_AL, fill_opacity=0.15, stroke_width=3), Text("validation\npick λ, epoch, arch", font_size=24, color=C_AL, line_spacing=1.1)),
            VGroup(Rectangle(width=2.8, height=1.4, stroke_color=C_BV, fill_color=C_BV, fill_opacity=0.15, stroke_width=3), Text("test\none final report", font_size=26, color=C_BV, line_spacing=1.1)),
        )
        for b in boxes:
            b[1].move_to(b[0])
        boxes.arrange(RIGHT, buff=0.35).shift(UP * 0.5)
        self.play(FadeIn(boxes), run_time=1.2)
        self.say("never tune on test · peeking turns test into a second val set", FG, 28)
        self.end()
        # 2 k-fold
        self.begin("kfold")
        self.play(FadeOut(boxes), run_time=0.5)
        folds = VGroup()
        for k in range(5):
            row = VGroup()
            for j in range(5):
                col = C_AL if j == k else C_GOOD
                lab = "val" if j == k else "tr"
                cell = VGroup(Rectangle(width=1.5, height=0.7, stroke_color=col, fill_color=col, fill_opacity=0.2, stroke_width=2),
                              Text(lab, font_size=22, color=col))
                cell[1].move_to(cell[0]); row.add(cell)
            row.arrange(RIGHT, buff=0.08)
            folds.add(row)
        folds.arrange(DOWN, buff=0.12).shift(UP * 0.2)
        self.play(FadeIn(folds), run_time=1.3)
        self.say("5-fold: each chunk is val once · average the 5 scores", FG, 28)
        self.end()
        # 3 leakage speakers
        self.begin("speak")
        self.play(FadeOut(folds), run_time=0.5)
        bad = VGroup(
            Text("LEAK (speech): same speaker in train and test", font_size=28, color=C_BAD),
            Text("model memorizes voice → WER looks great, fails on new callers", font_size=26, color=MUTED),
            Text("fix: split by speaker_id (or session), not by utterance", font_size=26, color=C_GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(UP * 0.6)
        self.play(FadeIn(bad), run_time=1.2)
        self.say("group leakage: correlated rows must stay on one side of the split", C_BAD, 26)
        self.end()
        # 4 normalize leak
        self.begin("norm")
        self.play(FadeOut(bad), run_time=0.5)
        n = VGroup(
            Text("LEAK: normalize with mean/std computed on train+test", font_size=28, color=C_BAD),
            Text("test statistics sneak into training → optimistic metrics", font_size=26, color=MUTED),
            Text("fix: fit scaler on train only; apply those numbers to val/test", font_size=26, color=C_GOOD),
            Text("also leaky: picking features with all-data correlation tests", font_size=26, color=C_DO),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).shift(UP * 0.3)
        self.play(FadeIn(n), run_time=1.2)
        self.say("rule: any choice that used test labels/stats is leakage", FG, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch5: Loss as MLE
class LossMLE(StepScene):
    def construct(self):
        title = caption("Where losses come from: maximum likelihood", 32).to_edge(UP, buff=0.35)
        # 1 idea
        self.begin("idea")
        self.play(FadeIn(title), run_time=0.7)
        idea = VGroup(
            Text("Assume a noise model for y | x, θ", font_size=30),
            Text("Likelihood = P(data | θ)   →   maximize it", font_size=30, color=C_LOSS),
            Text("≡ minimize  −log likelihood   (the loss)", font_size=30, color=C_LOSS),
        ).arrange(DOWN, buff=0.4).shift(UP * 0.4)
        self.play(FadeIn(idea), run_time=1.2)
        self.say("loss is not arbitrary — it encodes an assumption about noise", FG, 28)
        self.end()
        # 2 MSE
        self.begin("mse")
        self.play(FadeOut(idea), run_time=0.5)
        h = Text("Regression: Gaussian noise  y = fθ(x) + ε,  ε ~ N(0, σ²)", font_size=28, color=C_BV).shift(UP * 2.0)
        self.play(FadeIn(h), run_time=0.7)
        der = MathTex(r"-\log p(y\mid x)=-\log\Big(\tfrac{1}{\sqrt{2\pi}\sigma}e^{-(y-f)^2/(2\sigma^2)}\Big)", font_size=36).shift(UP * 0.7)
        self.play(Write(der), run_time=1.6)
        res = MathTex(r"= \tfrac{1}{2\sigma^2}(y-f_\theta(x))^2 + \mathrm{const}", font_size=40, color=C_GOOD).shift(DOWN * 0.6)
        self.play(Write(res), run_time=1.2)
        self.say("MSE = NLL under Gaussian noise (σ fixed)", C_GOOD, 28)
        self.end()
        # 3 CE bernoulli
        self.begin("ce")
        self.play(FadeOut(VGroup(h, der, res)), run_time=0.5)
        h2 = Text("Binary classification: y ~ Bernoulli(p),  p = σ(z)", font_size=28, color=C_SIG).shift(UP * 2.0)
        self.play(FadeIn(h2), run_time=0.7)
        lik = MathTex(r"p(y\mid p)=p^y(1-p)^{1-y}", font_size=44).shift(UP * 0.8)
        self.play(Write(lik), run_time=1.2)
        nll = MathTex(r"-\log p= -[y\log p+(1-y)\log(1-p)]", font_size=42, color=C_GOOD).shift(DOWN * 0.4)
        self.play(Write(nll), run_time=1.3)
        self.say("binary cross-entropy = Bernoulli NLL", C_GOOD, 28)
        self.end()
        # 4 categorical
        self.begin("cat")
        self.play(FadeOut(VGroup(h2, lik, nll)), run_time=0.5)
        h3 = Text("K classes: y one-hot, p = softmax(z)", font_size=28, color=C_DO).shift(UP * 1.6)
        ce = MathTex(r"L=-\sum_{k=1}^{K} y_k\log p_k", font_size=48).shift(UP * 0.3)
        note = Text("when y is one-hot this is just −log p_correct", font_size=28, color=MUTED).shift(DOWN * 0.8)
        self.play(FadeIn(h3), Write(ce), FadeIn(note), run_time=1.6)
        self.say("categorical CE = multinoulli NLL · same idea, K-way", FG, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch6: Gradient descent
class GradDesc(StepScene):
    def construct(self):
        title = caption("Gradient descent: follow −∇ to lower the loss", 32).to_edge(UP, buff=0.35)
        # 1 rule
        self.begin("rule")
        self.play(FadeIn(title), run_time=0.7)
        f = MathTex(r"w \leftarrow w - \eta\,\nabla_w L(w)", font_size=56).shift(UP * 1.2)
        self.play(Write(f), run_time=1.4)
        bits = VGroup(
            Text("∇L = direction of steepest ascent", font_size=28, color=MUTED),
            Text("minus sign → descend · η = learning rate (step size)", font_size=28, color=C_GD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).shift(DOWN * 0.4)
        self.play(FadeIn(bits), run_time=1.0)
        self.say("one step: evaluate gradient at current w, then nudge", FG, 28)
        self.end()
        # 2 lr
        self.begin("lr")
        self.play(FadeOut(VGroup(f, bits)), run_time=0.5)
        axes = Axes(x_range=[-0.5, 4.5, 1], y_range=[0, 5, 1], x_length=8, y_length=3.6,
                    axis_config={"color": MUTED}, tips=False).shift(DOWN * 0.2)
        curve = axes.plot(lambda x: (x - 3) ** 2, color=C_BV, stroke_width=4)
        self.play(Create(axes), Create(curve), run_time=1.0)
        # too big
        dots_big = [axes.c2p(0, 9)]  # will clip - use visible points
        # show annotations instead
        a = VGroup(
            Text("η too small → crawl", font_size=26, color=C_UN),
            Text("η too big → overshoot / diverge", font_size=26, color=C_BAD),
            Text("η just right → steady progress to the min", font_size=26, color=C_GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.4).shift(UP * 0.8)
        self.play(FadeIn(a), run_time=1.0)
        self.say("tune η on val loss; schedules (decay, warmup) help deep nets", FG, 26)
        self.end()
        # 3 by hand
        self.begin("hand")
        self.play(FadeOut(VGroup(axes, curve, a)), run_time=0.5)
        h = Text("By hand:  L(w)=(w−3)²   ·   ∇L=2(w−3)   ·   η=0.25   ·   w₀=0", font_size=26, color=C_AL).shift(UP * 2.1)
        self.play(FadeIn(h), run_time=0.7)
        steps = VGroup(
            Text("step 1:  g=2(0−3)=−6   →   w=0−0.25(−6)=1.5", font_size=28),
            Text("step 2:  g=2(1.5−3)=−3  →   w=1.5−0.25(−3)=2.25", font_size=28),
            Text("step 3:  g=2(2.25−3)=−1.5 →  w=2.25−0.25(−1.5)=2.625", font_size=28),
            Text("→ closing in on w*=3", font_size=28, color=C_GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(DOWN * 0.2)
        self.play(FadeIn(steps), run_time=1.4)
        self.say("write g first, then the update — don't skip the gradient", FG, 28)
        self.end()
        # 4 sgd
        self.begin("sgd")
        self.play(FadeOut(VGroup(h, steps)), run_time=0.5)
        tab = VGroup(
            Text("batch GD:  ∇ over all N examples — exact but slow / memory-heavy", font_size=26, color=C_BV),
            Text("SGD:       ∇ on 1 random example — noisy, cheap, often generalizes", font_size=26, color=C_DO),
            Text("mini-batch: ∇ on B examples (32–256) — GPU-friendly middle ground", font_size=26, color=C_GOOD),
            Text("noise in SGD can help escape sharp minima; batch size tunes it", font_size=26, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(UP * 0.3)
        self.play(FadeIn(tab), run_time=1.3)
        self.say("in practice almost everyone uses mini-batch SGD / Adam", C_GOOD, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch7: Logistic regression
class LogReg(StepScene):
    def construct(self):
        title = caption("Logistic regression: sigmoid, log-loss, gradient (p−y)x", 30).to_edge(UP, buff=0.35)
        # 1 sigmoid
        self.begin("sig")
        self.play(FadeIn(title), run_time=0.7)
        s = MathTex(r"p=\sigma(z)=\frac{1}{1+e^{-z}}\in(0,1)", font_size=52).shift(UP * 1.3)
        self.play(Write(s), run_time=1.4)
        props = VGroup(
            Text("z = w·x  (plus bias) · score → probability", font_size=28, color=MUTED),
            Text("σ(0)=0.5 · σ→1 as z→+∞ · σ→0 as z→−∞", font_size=28, color=C_SIG),
            Text("σ'(z)=σ(z)(1−σ(z))  — handy for backprop", font_size=28, color=C_SIG),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(DOWN * 0.5)
        self.play(FadeIn(props), run_time=1.1)
        self.say("one linear score, squashed to a probability", FG, 28)
        self.end()
        # 2 loss
        self.begin("loss")
        self.play(FadeOut(VGroup(s, props)), run_time=0.5)
        loss = MathTex(r"L=-[y\log p+(1-y)\log(1-p)]", font_size=48).shift(UP * 1.4)
        self.play(Write(loss), run_time=1.3)
        note = VGroup(
            Text("y=1 → L=−log p   (punish low confidence on positives)", font_size=26, color=C_GOOD),
            Text("y=0 → L=−log(1−p)", font_size=26, color=C_BAD),
            Text("same Bernoulli NLL from the last chapter", font_size=26, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(DOWN * 0.3)
        self.play(FadeIn(note), run_time=1.0)
        self.say("perfect prediction → L→0 · confident & wrong → L→∞", FG, 28)
        self.end()
        # 3 grad derive
        self.begin("grad")
        self.play(FadeOut(VGroup(loss, note)), run_time=0.5)
        d = VGroup(
            MathTex(r"\frac{\partial L}{\partial z}=p-y", font_size=48),
            Text("chain: ∂L/∂p · ∂p/∂z simplifies to (p − y)", font_size=26, color=MUTED),
            MathTex(r"\frac{\partial L}{\partial w}=(p-y)\,x", font_size=48, color=C_AL),
            Text("because ∂z/∂w = x ·  (vector form: (p−y)x)", font_size=26, color=MUTED),
        ).arrange(DOWN, buff=0.35).shift(UP * 0.2)
        self.play(FadeIn(d), run_time=1.5)
        self.say("the famous (p−y)x gradient — remember this for interviews", C_AL, 28)
        self.end()
        # 4 by hand
        self.begin("hand")
        self.play(FadeOut(d), run_time=0.5)
        setup = Text("x=2,  w=0.3,  y=1,  η=0.5", font_size=28, color=C_AL).shift(UP * 2.1)
        self.play(FadeIn(setup), run_time=0.6)
        steps = VGroup(
            Text("z = w·x = 0.6", font_size=28),
            Text("p = σ(0.6) = 1/(1+e^{−0.6}) ≈ 0.646", font_size=28),
            Text("L = −log(0.646) ≈ 0.437", font_size=28),
            Text("∇w = (p−y)x = (0.646−1)·2 ≈ −0.709", font_size=28, color=C_AL),
            Text("w ← 0.3 − 0.5(−0.709) ≈ 0.654", font_size=28, color=C_GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(DOWN * 0.15)
        self.play(FadeIn(steps), run_time=1.5)
        self.say("p < y → gradient negative → w increases → p rises next time", C_GOOD, 26)
        self.end(2.2)

# ---------------------------------------------------------------- Ch8: Backprop
class Backprop(StepScene):
    def construct(self):
        title = caption("Backprop by hand on a tiny net", 32).to_edge(UP, buff=0.35)
        # 1 net
        self.begin("net")
        self.play(FadeIn(title), run_time=0.7)
        net = MathTex(r"x \xrightarrow{w_1} z=w_1 x \xrightarrow{\sigma} h=\sigma(z) \xrightarrow{w_2} \hat y=w_2 h", font_size=36).shift(UP * 1.5)
        self.play(Write(net), run_time=1.6)
        loss = MathTex(r"L=\tfrac12(\hat y-y)^2", font_size=44, color=C_LOSS).shift(UP * 0.3)
        vals = Text("numbers: x=1, w₁=1, w₂=2, y=1   (no bias, for clarity)", font_size=26, color=MUTED).shift(DOWN * 0.6)
        self.play(Write(loss), FadeIn(vals), run_time=1.2)
        self.say("1 input → 1 sigmoid hidden → linear output · squared loss", FG, 28)
        self.end()
        # 2 forward
        self.begin("fwd")
        self.play(FadeOut(VGroup(net, loss, vals)), run_time=0.5)
        fwd = VGroup(
            Text("Forward pass", font_size=30, color=C_GOOD),
            Text("z = w₁·x = 1·1 = 1", font_size=28),
            Text("h = σ(1) = 1/(1+e^{−1}) ≈ 0.731", font_size=28),
            Text("ŷ = w₂·h = 2·0.731 ≈ 1.462", font_size=28),
            Text("L = ½(1.462−1)² ≈ 0.107", font_size=28, color=C_LOSS),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(UP * 0.3)
        self.play(FadeIn(fwd), run_time=1.4)
        self.say("write every intermediate — you'll need them going backward", FG, 28)
        self.end()
        # 3 backward
        self.begin("bwd")
        self.play(FadeOut(fwd), run_time=0.5)
        bwd = VGroup(
            Text("Backward (chain rule)", font_size=30, color=C_BP),
            Text("∂L/∂ŷ = ŷ−y ≈ 0.462", font_size=28),
            Text("∂L/∂w₂ = (∂L/∂ŷ)·h ≈ 0.462·0.731 ≈ 0.338", font_size=28, color=C_AL),
            Text("∂L/∂h = (∂L/∂ŷ)·w₂ ≈ 0.462·2 ≈ 0.924", font_size=28),
            Text("∂h/∂z = h(1−h) ≈ 0.731·0.269 ≈ 0.197", font_size=28),
            Text("∂L/∂w₁ = (∂L/∂h)·(∂h/∂z)·x ≈ 0.924·0.197·1 ≈ 0.182", font_size=28, color=C_AL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).shift(UP * 0.15)
        self.play(FadeIn(bwd), run_time=1.6)
        self.say("each local derivative multiplies along the path to that weight", C_BP, 26)
        self.end()
        # 4 update + faded
        self.begin("upd")
        self.play(FadeOut(bwd), run_time=0.5)
        upd = VGroup(
            Text("One update with η=0.5", font_size=30, color=C_GOOD),
            Text("w₂ ← 2 − 0.5·0.338 ≈ 1.831", font_size=28),
            Text("w₁ ← 1 − 0.5·0.182 ≈ 0.909", font_size=28),
            Text("both fell: ŷ was too high, so we reduce the path weights", font_size=26, color=MUTED),
            Text("Faded: if y=0 instead, ∂L/∂ŷ=ŷ−0>0 still… wait ŷ≈1.46,", font_size=24, color=C_DO),
            Text("sign of (ŷ−y) flips the whole chain — check it on paper", font_size=24, color=C_DO),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(UP * 0.2)
        self.play(FadeIn(upd), run_time=1.5)
        self.say("interview move: forward numbers → chain → one update", C_GOOD, 28)
        self.end(2.2)
