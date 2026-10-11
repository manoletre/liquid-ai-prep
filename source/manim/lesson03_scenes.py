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

C_TP = "#9ece6a"; C_FP = "#ff9e64"; C_FN = "#f7768e"; C_TN = "#7aa2f7"
C_P = "#b392f0"; C_RC = "#7dcfff"; C_F1 = "#e0af68"; C_T = "#e08fd0"

def frac(num, den, val, col=FG, fs=44):
    t = MathTex(num, r"\over", den, r"=", val, font_size=fs)
    return t

def matrix(vals, cell=1.55, fs=40):
    """2x2 confusion matrix; vals = dict TP, FN, FP, TN (strings). Returns VGroup with .cells dict."""
    g = VGroup(); cells = {}
    pos = {"TP": (0, 0), "FN": (0, 1), "FP": (1, 0), "TN": (1, 1)}
    col = {"TP": C_TP, "FN": C_FN, "FP": C_FP, "TN": C_TN}
    for k, (r, c) in pos.items():
        sq = Square(cell, stroke_color=col[k], stroke_width=3, fill_color=col[k], fill_opacity=0.12).move_to([c * cell, -r * cell, 0])
        nm = Text(k, font_size=24, color=col[k]).move_to(sq.get_top() + DOWN * 0.28)
        v = Text(vals.get(k, ""), font_size=fs, color=FG).move_to(sq.get_center() + DOWN * 0.12)
        cells[k] = VGroup(sq, nm, v); g.add(cells[k])
    ch = VGroup(Text("flagged", font_size=22, color=MUTED).next_to(cells["TP"], UP, buff=0.15),
                Text("passed", font_size=22, color=MUTED).next_to(cells["FN"], UP, buff=0.15))
    rh = VGroup(Text("actually +", font_size=22, color=MUTED).next_to(cells["TP"], LEFT, buff=0.15),
                Text("actually −", font_size=22, color=MUTED).next_to(cells["FP"], LEFT, buff=0.15))
    g.add(ch, rh); g.cells = cells
    return g

# ---------------------------------------------------------------- L3 Ch1
class Confusion(StepScene):
    def construct(self):
        title = caption("Hallucination detector on 1,000 LFM2 outputs", 34).to_edge(UP, buff=0.35)
        sq = VGroup(*[Square(0.34, stroke_width=0, fill_color=GRID, fill_opacity=1) for _ in range(100)]).arrange_in_grid(10, 10, buff=0.07)
        sq.move_to(LEFT * 3.6 + DOWN * 0.3)
        pos = list(range(10)); flagged_pos = pos[:8]; flagged_neg = [10, 11, 12, 13]
        # 1
        self.begin("setup")
        self.play(FadeIn(title), FadeIn(sq), run_time=1)
        self.play(*[sq[i].animate.set_fill("#c8ccd6") for i in pos], run_time=1.2)
        leg = Text("1 square = 10 outputs", font_size=24, color=MUTED).next_to(sq, DOWN, buff=0.25)
        self.play(FadeIn(leg), run_time=0.6)
        self.say("100 hallucinated (positive, light) · 900 fine (negative, dark)", FG, 30)
        self.end()
        # 2
        self.begin("flags")
        self.play(*[sq[i].animate.set_stroke(C_T, 4) for i in flagged_pos + flagged_neg], run_time=1.4)
        self.say("the detector flags 120 outputs (pink outline)", C_T)
        self.end()
        m = matrix({}).scale(0.95).move_to(RIGHT * 3.4 + UP * 0.55)
        # 3
        self.begin("tp_fn")
        self.play(FadeIn(m), run_time=1)
        self.play(*[sq[i].animate.set_fill(C_TP) for i in flagged_pos], run_time=1)
        tp = Text("80", font_size=40).move_to(m.cells["TP"][0].get_center() + DOWN * 0.12); self.play(FadeIn(tp), run_time=0.6)
        self.play(*[sq[i].animate.set_fill(C_FN) for i in (8, 9)], run_time=1)
        fn = Text("20", font_size=40).move_to(m.cells["FN"][0].get_center() + DOWN * 0.12); self.play(FadeIn(fn), run_time=0.6)
        self.say("of 100 hallucinations: 80 caught (TP), 20 missed (FN)", FG, 30)
        self.end()
        # 4
        self.begin("fp_tn")
        self.play(*[sq[i].animate.set_fill(C_FP) for i in flagged_neg], run_time=1)
        fp = Text("40", font_size=40).move_to(m.cells["FP"][0].get_center() + DOWN * 0.12); self.play(FadeIn(fp), run_time=0.6)
        self.play(*[sq[i].animate.set_fill(C_TN).set_opacity(0.55) for i in range(14, 100)], run_time=1.2)
        tn = Text("860", font_size=40).move_to(m.cells["TN"][0].get_center() + DOWN * 0.12); self.play(FadeIn(tn), run_time=0.6)
        self.say("of 900 fine outputs: 40 false alarms (FP), 860 correctly passed (TN)", FG, 28)
        self.end()
        # 5
        self.begin("read")
        rows = VGroup(Text("TP  flagged & really hallucinated", font_size=24, color=C_TP),
                      Text("FP  false alarm", font_size=24, color=C_FP),
                      Text("FN  missed hallucination", font_size=24, color=C_FN),
                      Text("TN  fine & passed", font_size=24, color=C_TN)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        rows.next_to(m, DOWN, buff=0.3).align_to(m, LEFT)
        self.play(FadeOut(leg), FadeIn(rows, shift=UP * 0.1), run_time=1)
        self.say("rows = truth · columns = prediction · total 1,000", FG, 30)
        self.end(2.2)

# ---------------------------------------------------------------- L3 Ch2
class Accuracy(StepScene):
    def construct(self):
        f = MathTex(r"\text{accuracy}", r"=", r"{TP", r"+", r"TN", r"\over", r"TP+FP+FN+TN}", font_size=56).to_edge(UP, buff=0.5)
        f[2].set_color(C_TP); f[4].set_color(C_TN)
        # 1
        self.begin("formula")
        self.play(Write(f), run_time=2)
        w = MathTex(r"{80+860 \over 1000}", r"=", r"94\%", font_size=52).next_to(f, DOWN, buff=0.6)
        w[2].set_color(C_F1)
        self.play(Write(w), run_time=1.5)
        self.say("detector from chapter 1: 94% of outputs classified correctly")
        self.end()
        # 2
        self.begin("imbalance")
        self.play(FadeOut(w), run_time=0.5)
        t = Text("10,000 support calls · only 100 must be escalated (1%)", font_size=30).shift(UP * 0.9)
        bneg = Rectangle(width=11.88, height=0.6, stroke_width=0, fill_color=C_TN, fill_opacity=0.6)
        bpos = Rectangle(width=0.12, height=0.6, stroke_width=0, fill_color=C_FN, fill_opacity=1)
        bar = VGroup(bneg, bpos).arrange(RIGHT, buff=0).shift(DOWN * 0.1)
        l1 = Text("9,900 routine", font_size=26, color=C_TN).next_to(bneg, DOWN, buff=0.15)
        l2 = Text("100 escalate", font_size=26, color=C_FN).next_to(bpos, DOWN, buff=0.15).shift(LEFT * 0.8)
        self.play(FadeIn(t), FadeIn(bar), FadeIn(l1), FadeIn(l2), run_time=1.5)
        self.say("heavy class imbalance: 99 / 1")
        self.end()
        # 3
        self.begin("lazy")
        self.play(FadeOut(VGroup(t, bar, l1, l2)), run_time=0.6)
        m = matrix({"TP": "0", "FN": "100", "FP": "0", "TN": "9,900"}, fs=34).scale(0.9).move_to(LEFT * 2.6 + DOWN * 0.7)
        lab = Text("model: “never escalate”", font_size=30, color=C_AL).next_to(m, UP, buff=0.65)
        self.play(FadeIn(lab), FadeIn(m), run_time=1.2)
        acc = MathTex(r"{0+9900 \over 10000}", r"=", r"99\%", font_size=50).move_to(RIGHT * 3.6 + DOWN * 0.3)
        acc[2].set_color(C_GOOD)
        self.play(Write(acc), run_time=1.5)
        self.say("99% accuracy without learning anything", C_AL)
        self.end()
        # 4
        self.begin("recall0")
        rc = MathTex(r"\text{caught }", r"0", r"\text{ of }", r"100", r"\text{ escalations}", font_size=44).next_to(acc, DOWN, buff=0.7)
        rc[1].set_color(C_FN)
        self.play(Write(rc), Indicate(m.cells["FN"], color=C_FN), run_time=1.6)
        self.say("accuracy is dominated by the majority class: look at the positives", C_BAD, 28)
        self.end(2.2)

# ---------------------------------------------------------------- L3 Ch3
POS = [0.95, 0.85, 0.70, 0.55, 0.30]; NEG = [0.80, 0.60, 0.40, 0.20, 0.10]

class PrecRecall(StepScene):
    def construct(self):
        pf = MathTex(r"\text{precision}", r"=", r"{TP", r"\over", r"TP+FP}", font_size=46)
        rf = MathTex(r"\text{recall}", r"=", r"{TP", r"\over", r"TP+FN}", font_size=46)
        pf[0].set_color(C_P); pf[2].set_color(C_TP); rf[0].set_color(C_RC); rf[2].set_color(C_TP)
        pf[4][3:].set_color(C_FP); rf[4][3:].set_color(C_FN)
        pf.move_to(LEFT * 3.4 + UP * 2.9); rf.move_to(RIGHT * 3.4 + UP * 2.9)
        # 1
        self.begin("formulas")
        self.play(Write(pf), run_time=1.5)
        t1 = Text("of what I flagged,\nhow much was real?", font_size=26, color=C_P, line_spacing=1.1).next_to(pf, DOWN, buff=0.25)
        self.play(FadeIn(t1), run_time=0.6); self.wait(0.5)
        self.play(Write(rf), run_time=1.5)
        t2 = Text("of what was really there,\nhow much did I catch?", font_size=26, color=C_RC, line_spacing=1.1).next_to(rf, DOWN, buff=0.25)
        self.play(FadeIn(t2), run_time=0.6)
        w = MathTex(r"\text{ch. 1:}\quad", r"P={80\over120}=0.67", r"\qquad", r"R={80\over100}=0.80", font_size=42).shift(DOWN * 1.3)
        w[1].set_color(C_P); w[3].set_color(C_RC)
        self.play(Write(w), run_time=1.5)
        self.end()
        # 2
        self.begin("scores")
        self.play(FadeOut(VGroup(t1, t2, w)), run_time=0.6)
        nl = NumberLine(x_range=[0, 1, 0.1], length=11, include_numbers=True, font_size=24, color=MUTED, decimal_number_config={"num_decimal_places": 1}).shift(DOWN * 0.6)
        lab = Text("model score", font_size=24, color=MUTED).next_to(nl, DOWN, buff=0.5)
        dots = {}
        for s in POS:
            d = VGroup(Circle(0.2, color=FG, stroke_width=3, fill_opacity=0), Text("+", font_size=26)).move_to(nl.n2p(s) + UP * 0.75); dots[("p", s)] = d
        for s in NEG:
            d = VGroup(Circle(0.2, color=MUTED, stroke_width=3, fill_opacity=0), Text("−", font_size=26, color=MUTED)).move_to(nl.n2p(s) + UP * 1.35); dots[("n", s)] = d
        self.play(Create(nl), FadeIn(lab), run_time=1)
        self.play(*[FadeIn(d) for d in dots.values()], run_time=1.2)
        self.say("5 positives (+, low row) · 5 negatives (−, high row)", FG, 30)
        self.end()
        tl = ValueTracker(0.5)
        line = always_redraw(lambda: DashedLine(nl.n2p(tl.get_value()) + DOWN * 0.15, nl.n2p(tl.get_value()) + UP * 1.8, color=C_T, stroke_width=4))
        tlab = always_redraw(lambda: MathTex(rf"t={tl.get_value():.2f}", color=C_T, font_size=36).next_to(nl.n2p(tl.get_value()) + UP * 1.8, UP, buff=0.1))
        def color_at(t):
            anims = []
            for (k, s), d in dots.items():
                if k == "p": c = C_TP if s >= t else C_FN
                else: c = C_FP if s >= t else C_TN
                anims += [d[0].animate.set_stroke(c).set_fill(c, opacity=0.35), d[1].animate.set_color(c)]
            return anims
        def pr(t):
            tp = sum(s >= t for s in POS); fp = sum(s >= t for s in NEG); fn = 5 - tp
            P = tp / (tp + fp) if tp + fp else 1.0; R = tp / 5
            P = int(P * 100 + 0.5) / 100
            g = MathTex(rf"P={{{tp}\over{tp+fp}}}={P:.2f}", r"\qquad", rf"R={{{tp}\over 5}}={R:.2f}", font_size=40).shift(DOWN * 2.35)
            g[0].set_color(C_P); g[2].set_color(C_RC); return g
        # 3
        self.begin("t05")
        self.add(line, tlab)
        self.play(FadeIn(line), run_time=0.6)
        self.play(*color_at(0.5), run_time=1.2)
        g = pr(0.5); self.unsay(); self.play(Write(g), run_time=1.2)
        self.cap = None
        self.end()
        # 4
        self.begin("t09")
        self.play(tl.animate.set_value(0.9), *color_at(0.9), run_time=2)
        g2 = pr(0.9); self.play(FadeTransform(g, g2), run_time=1); g = g2
        self.end()
        # 5
        self.begin("t025")
        self.play(tl.animate.set_value(0.25), *color_at(0.25), run_time=2.4)
        g2 = pr(0.25); self.play(FadeTransform(g, g2), run_time=1); g = g2
        self.end()
        # 6
        self.begin("which")
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        self.remove(line, tlab)
        rule = Text("raise t → precision ↑, recall ↓   ·   lower t → recall ↑, precision ↓", font_size=28, color=C_T).to_edge(UP, buff=0.6)
        rows = VGroup(
            Text("spam filter → precision  (don't bin real email)", font_size=30, color=C_P),
            Text("medical screening → recall  (don't miss disease)", font_size=30, color=C_RC),
            Text("fraud → recall, with a precision floor (review cost)", font_size=30, color=C_RC),
            Text("call escalation → recall  (a missed escalation is costly)", font_size=30, color=C_RC),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).shift(DOWN * 0.3)
        self.play(FadeIn(rule), run_time=1)
        for r in rows:
            self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.8); self.wait(0.3)
        self.say("ask: which mistake is more expensive, FP or FN?", C_AL)
        self.end(2.2)

# ---------------------------------------------------------------- L3 Ch4
class F1Score(StepScene):
    def construct(self):
        # 1
        self.begin("why")
        t = Text("One number from P and R. Example: P = 1.0, R = 0.2", font_size=32).to_edge(UP, buff=0.5)
        self.play(FadeIn(t), run_time=0.8)
        ax = Axes(x_range=[0, 4, 1], y_range=[0, 1, 0.25], x_length=8, y_length=4, axis_config={"color": MUTED, "include_tip": False},
                  y_axis_config={"numbers_to_include": [0.25, 0.5, 0.75, 1.0], "font_size": 24}).shift(DOWN * 0.6)
        vals = [("P", 1.0, C_P), ("R", 0.2, C_RC), ("mean", 0.6, MUTED), ("F1", 1 / 3, C_F1)]
        bars = VGroup()
        for i, (n, v, c) in enumerate(vals):
            r = Rectangle(width=1.2, height=ax.c2p(0, v)[1] - ax.c2p(0, 0)[1], stroke_width=0, fill_color=c, fill_opacity=0.9).move_to(ax.c2p(i + 0.5, 0), aligned_edge=DOWN)
            lab = Text(n, font_size=26, color=c).next_to(ax.c2p(i + 0.5, 0), DOWN, buff=0.2)
            num = Text(f"{v:.2f}", font_size=26).next_to(r, UP, buff=0.08)
            bars.add(VGroup(r, lab, num))
        self.play(Create(ax), FadeIn(bars[0]), FadeIn(bars[1]), run_time=1.2)
        self.wait(0.6)
        self.play(FadeIn(bars[2]), run_time=0.8)
        self.say("arithmetic mean 0.60 looks OK, but recall is terrible", C_AL); self.wait(0.6)
        self.play(FadeIn(bars[3]), run_time=0.8)
        self.say("harmonic mean (F1) = 0.33: pulled toward the weaker one", C_F1)
        self.end(2.2)
        # 2
        self.begin("formula")
        self.unsay(); self.play(FadeOut(VGroup(t, ax, bars)), run_time=0.7)
        f = MathTex(r"F_1", r"=", r"\frac{2}{\frac{1}{P}+\frac{1}{R}}", font_size=56).shift(UP * 1.5)
        f[0].set_color(C_F1)
        self.play(Write(f), run_time=1.6)
        f2 = MathTex(r"=", r"{2PR \over P+R}", r"=", r"{2\,TP \over 2\,TP+FP+FN}", font_size=52).next_to(f, DOWN, buff=0.5)
        f2[3][0:3].set_color(C_TP)
        self.play(Write(f2), run_time=2)
        self.say("harmonic mean: averages the reciprocals")
        self.end()
        # 3
        self.begin("worked")
        self.play(FadeOut(f), FadeOut(f2), run_time=0.6); self.unsay()
        w1 = MathTex(r"\text{ch. 1: }P=0.667,\ R=0.80", font_size=44).shift(UP * 1.6)
        w2 = MathTex(r"F_1={2\cdot0.667\cdot0.80 \over 0.667+0.80}", r"=", r"0.727", font_size=50).next_to(w1, DOWN, buff=0.6)
        w3 = MathTex(r"\text{check: }{2\cdot80 \over 2\cdot80+40+20}={160\over220}=0.727", font_size=42).next_to(w2, DOWN, buff=0.6)
        w2[2].set_color(C_F1)
        self.play(Write(w1), run_time=1.2); self.play(Write(w2), run_time=1.8); self.play(Write(w3), run_time=1.8)
        self.end()
        # 4
        self.begin("fbeta")
        self.play(FadeOut(VGroup(w1, w2, w3)), run_time=0.6)
        fb = MathTex(r"F_\beta", r"=", r"{(1+\beta^2)\,P\,R \over \beta^2 P + R}", font_size=54).shift(UP * 1.6)
        fb[0].set_color(C_F1)
        self.play(Write(fb), run_time=1.8)
        r1 = MathTex(r"\beta=2:\ F_2=0.769", r"\quad\text{(recall counts more)}", font_size=40).next_to(fb, DOWN, buff=0.7)
        r2 = MathTex(r"\beta=0.5:\ F_{0.5}=0.690", r"\quad\text{(precision counts more)}", font_size=40).next_to(r1, DOWN, buff=0.4)
        r1[0].set_color(C_RC); r2[0].set_color(C_P)
        self.play(Write(r1), run_time=1.4); self.play(Write(r2), run_time=1.4)
        self.say("same detector (P = 0.667, R = 0.80); β = 1 gives back F1")
        self.end(2.2)

# ---------------------------------------------------------------- L3 Ch5
class RocPr(StepScene):
    def construct(self):
        # 1
        self.begin("rates")
        tpr = MathTex(r"\text{TPR}", r"=", r"{TP \over TP+FN}", r"=\text{recall}", font_size=46)
        fpr = MathTex(r"\text{FPR}", r"=", r"{FP \over FP+TN}", font_size=46)
        tpr[0].set_color(C_RC); fpr[0].set_color(C_FP)
        VGroup(tpr, fpr).arrange(DOWN, buff=0.6, aligned_edge=LEFT).shift(UP * 0.6)
        self.play(Write(tpr), run_time=1.5); self.play(Write(fpr), run_time=1.5)
        self.say("share of positives caught vs share of negatives falsely flagged")
        self.end()
        # 2
        self.begin("sweep")
        self.play(FadeOut(tpr), FadeOut(fpr), run_time=0.6); self.unsay()
        ax = Axes(x_range=[0, 1, 0.2], y_range=[0, 1, 0.2], x_length=5.4, y_length=5.0, axis_config={"color": MUTED, "include_tip": False, "font_size": 22},
                  x_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8, 1.0]}, y_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8, 1.0]}).shift(LEFT * 2.6 + UP * 0.1)
        xl = Text("FPR", font_size=26, color=C_FP).next_to(ax.x_axis, DOWN, buff=0.45)
        yl = Text("TPR", font_size=26, color=C_RC).next_to(ax.y_axis, UP, buff=0.15)
        diag = DashedLine(ax.c2p(0, 0), ax.c2p(1, 1), color=MUTED)
        dl = Text("random", font_size=22, color=MUTED).move_to(ax.c2p(0.72, 0.6)).rotate(PI / 4)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), Create(diag), FadeIn(dl), run_time=1.4)
        seq = sorted([(s, 1) for s in POS] + [(s, 0) for s in NEG], reverse=True)
        pts = [(0.0, 0.0)]; tp = fp = 0
        for s, y in seq:
            tp += y; fp += 1 - y; pts.append((fp / 5, tp / 5))
        info = Text("sweep the threshold from high to low:\nevery + moves the curve up,\nevery − moves it right", font_size=26, color=FG, line_spacing=1.2).move_to(RIGHT * 3.6 + UP * 0.8)
        self.play(FadeIn(info), run_time=0.8)
        curve = VGroup()
        for a, b in zip(pts, pts[1:]):
            seg = Line(ax.c2p(*a), ax.c2p(*b), color=C_RC, stroke_width=5); curve.add(seg)
            self.play(Create(seg), run_time=0.45)
        self.end()
        # 3
        self.begin("auc")
        area = VGroup()
        for a, b in zip(pts, pts[1:]):
            if b[0] > a[0]:
                area.add(Polygon(ax.c2p(a[0], 0), ax.c2p(b[0], 0), ax.c2p(b[0], b[1]), ax.c2p(a[0], a[1]), stroke_width=0, fill_color=C_RC, fill_opacity=0.25))
        self.play(FadeIn(area), FadeOut(info), run_time=1.2)
        a1 = MathTex(r"\text{AUROC}=0.76", font_size=48, color=C_RC).move_to(RIGHT * 3.6 + UP * 1.4)
        a2 = Text("= P(random + scores above random −)", font_size=24).next_to(a1, DOWN, buff=0.35)
        a3 = MathTex(r"{19 \text{ of } 25 \text{ pairs}}", font_size=40).next_to(a2, DOWN, buff=0.35)
        a4 = Text("0.5 = random · 1.0 = perfect ranking", font_size=24, color=MUTED).next_to(a3, DOWN, buff=0.45)
        self.play(Write(a1), run_time=1); self.play(FadeIn(a2), run_time=0.8); self.play(Write(a3), run_time=1); self.play(FadeIn(a4), run_time=0.8)
        self.end(2.2)
        # 4
        self.begin("imbalance")
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        t = Text("Rare positives: 1% of 10,000 (fraud, escalations)", font_size=32).to_edge(UP, buff=0.6)
        l1 = MathTex(r"\text{ROC point: }", r"\text{TPR}=0.90", r",\ ", r"\text{FPR}=0.05", r"\quad\text{(looks great)}", font_size=42).shift(UP * 1.2)
        l1[1].set_color(C_RC); l1[3].set_color(C_FP)
        l2 = MathTex(r"TP = 0.90\times100 = 90", r"\qquad", r"FP = 0.05\times 9900 = 495", font_size=42).next_to(l1, DOWN, buff=0.6)
        l2[0].set_color(C_TP); l2[2].set_color(C_FP)
        l3 = MathTex(r"\text{precision}={90 \over 90+495}", r"=", r"0.15", font_size=48).next_to(l2, DOWN, buff=0.6)
        l3[2].set_color(C_BAD)
        self.play(FadeIn(t), run_time=0.8); self.play(Write(l1), run_time=1.5); self.play(Write(l2), run_time=1.8); self.play(Write(l3), run_time=1.5)
        self.say("FPR hides 495 false alarms behind 9,900 negatives", C_BAD)
        self.end(2.2)
        # 5
        self.begin("pr")
        self.unsay(); self.play(FadeOut(VGroup(t, l1, l2, l3)), run_time=0.7)
        ax2 = Axes(x_range=[0, 1, 0.2], y_range=[0, 1, 0.2], x_length=5.4, y_length=4.5, axis_config={"color": MUTED, "include_tip": False, "font_size": 22},
                   x_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8, 1.0]}, y_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8, 1.0]}).shift(LEFT * 2.6 + UP * 0.4)
        xl2 = Text("recall", font_size=26, color=C_RC).next_to(ax2.x_axis, DOWN, buff=0.45)
        yl2 = Text("precision", font_size=26, color=C_P).next_to(ax2.y_axis, UP, buff=0.2)
        self.play(Create(ax2), FadeIn(xl2), FadeIn(yl2), run_time=1.2)
        prp = []; tp = fp = 0
        for s, y in seq:
            tp += y; fp += 1 - y; prp.append((tp / 5, tp / (tp + fp)))
        dots = VGroup(*[Dot(ax2.c2p(*p), color=C_P, radius=0.07) for p in prp])
        path = VMobject(color=C_P, stroke_width=4).set_points_as_corners([ax2.c2p(*p) for p in prp])
        base = DashedLine(ax2.c2p(0, 0.5), ax2.c2p(1, 0.5), color=MUTED)
        bl = Text("baseline = prevalence (0.5 here)", font_size=22, color=MUTED).next_to(ax2.c2p(0.5, 0.5), DOWN, buff=0.15)
        self.play(Create(path), FadeIn(dots), run_time=2)
        self.play(Create(base), FadeIn(bl), run_time=1)
        txt = Text("PR curve: precision at every recall.\nRandom = prevalence (1% for rare\nevents), so PR-AUC exposes\nfalse alarms that ROC hides.", font_size=26, line_spacing=1.2).move_to(RIGHT * 3.7)
        self.play(FadeIn(txt), run_time=1)
        self.say("rare positives → prefer PR-AUC", C_P, 30)
        self.end(2.4)

# ---------------------------------------------------------------- L3 Ch6
class WER(StepScene):
    def construct(self):
        f = MathTex(r"\text{WER}", r"=", r"{S", r"+", r"D", r"+", r"I", r"\over", r"N}", font_size=60).to_edge(UP, buff=0.5)
        f[0].set_color(C_F1); f[2].set_color(C_FP); f[4].set_color(C_FN); f[6].set_color(C_RC)
        # 1
        self.begin("formula")
        self.play(Write(f), run_time=1.8)
        leg = VGroup(Text("S substitutions (wrong word)", font_size=28, color=C_FP), Text("D deletions (missing word)", font_size=28, color=C_FN),
                     Text("I insertions (extra word)", font_size=28, color=C_RC), Text("N words in the reference", font_size=28, color=FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(DOWN * 0.6)
        for l in leg:
            self.play(FadeIn(l, shift=RIGHT * 0.2), run_time=0.6)
        self.end()
        # 2
        self.begin("align")
        self.play(FadeOut(leg), run_time=0.6)
        ref = ["i", "need", "to", "refill", "my", "prescription", ""]
        hyp = ["", "need", "to", "refill", "the", "prescription", "today"]
        ops = ["D", "", "", "", "S", "", "I"]
        cells = VGroup(); xs = []
        for i, (r, h, o) in enumerate(zip(ref, hyp, ops)):
            col = {"D": C_FN, "S": C_FP, "I": C_RC}.get(o, FG)
            rt = Text(r if r else "—", font_size=30, color=FG if not o else col)
            ht = Text(h if h else "—", font_size=30, color=FG if not o else col)
            cells.add(VGroup(rt, ht))
        widths = [max(c[0].width, c[1].width) + 0.55 for c in cells]
        x = -sum(widths) / 2 + 1.0
        for c, w in zip(cells, widths):
            c[0].move_to([x + w / 2, 0.6, 0]); c[1].move_to([x + w / 2, -0.4, 0]); x += w
        rl = Text("reference", font_size=24, color=MUTED).next_to(cells[0][0], LEFT, buff=0.5)
        hl = Text("ASR output", font_size=24, color=MUTED).next_to(cells[0][1], LEFT, buff=0.5)
        self.play(FadeIn(rl), FadeIn(hl), *[FadeIn(c[0]) for c in cells], run_time=1.2)
        self.play(*[FadeIn(c[1]) for c in cells], run_time=1.2)
        self.say("align the words (minimum word edit distance)")
        self.end()
        # 3
        self.begin("count")
        tags = VGroup()
        for i, o in enumerate(ops):
            if o:
                col = {"D": C_FN, "S": C_FP, "I": C_RC}[o]
                box = SurroundingRectangle(cells[i], color=col, buff=0.12)
                tg = Text(o, font_size=30, color=col).next_to(box, DOWN, buff=0.15)
                tags.add(box, tg)
                self.play(Create(box), FadeIn(tg), run_time=0.8)
        res = MathTex(r"\text{WER}=", r"{1", r"+", r"1", r"+", r"1", r"\over", r"6}", r"=", r"50\%", font_size=54).shift(DOWN * 2.2)
        res[1].set_color(C_FP); res[3].set_color(C_FN); res[5].set_color(C_RC); res[9].set_color(C_F1)
        self.unsay(); self.play(Write(res), run_time=1.6)
        self.end(2.2)
        # 4
        self.begin("notes")
        self.play(*[FadeOut(m) for m in self.mobjects if m is not f], run_time=0.8)
        rows = VGroup(
            Text("• N counts reference words only, so insertions can push WER above 100%", font_size=27),
            Text("• normalize first: case, punctuation, “10” vs “ten”", font_size=27),
            Text("• one wrong word can matter far more than the score shows:", font_size=27),
            Text("   “take 15 mg” → “take 50 mg” is still just 1 substitution", font_size=27, color=C_FP),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).shift(DOWN * 0.4)
        for r in rows:
            self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.8); self.wait(0.3)
        self.end(2.4)
