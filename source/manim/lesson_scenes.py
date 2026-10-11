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

C_ENT = "#7dcfff"

def group_bars(ax, vals, color_by_sign=False, fs=28, fmt="{:.2f}"):
    g = VGroup()
    for i, v in enumerate(vals):
        c = C_R if not color_by_sign else (C_GOOD if v > 0.05 else C_BAD if v < -0.05 else MUTED)
        h = max(0.03, abs(ax.c2p(0, v)[1] - ax.c2p(0, 0)[1]))
        rct = Rectangle(width=0.8, height=h, stroke_width=0, fill_color=c, fill_opacity=0.9)
        rct.move_to(ax.c2p(i + 0.5, 0), aligned_edge=DOWN if v >= 0 else UP)
        lab = Text(fmt.format(v).replace("-", "−"), font_size=fs, color=FG).next_to(rct, UP if v >= 0 else DOWN, buff=0.08)
        g.add(VGroup(rct, lab))
    return g

def stats(vals):
    a = np.array(vals, float); return a.mean(), a.std(ddof=1)

# ---------------------------------------------------------------- L2 Chapter 1
class GroupSignal(StepScene):
    def construct(self):
        f = MathTex(r"A_i", r"=", r"{r_i", r"-", r"\mu", r"\over", r"\sigma}", font_size=58)
        f[0].set_color(C_R); f[2].set_color(C_R); f[4].set_color(C_ST); f[6].set_color(C_ST)
        f.to_edge(UP, buff=0.4)
        ax = Axes(x_range=[0, 8, 1], y_range=[-1.6, 1.6, 0.5], x_length=12, y_length=4.4).shift(DOWN * 0.5)
        zero = Line(ax.c2p(0, 0), ax.c2p(8, 0), color=MUTED)
        names = VGroup(*[MathTex(f"y_{i+1}", font_size=36, color=C_Y).move_to(ax.c2p(i + 0.5, -1.82)) for i in range(8)])
        div = [0.90, 0.85, 0.88, 0.20, 0.10, 0.15, 0.12, 0.87]
        m, s = stats(div)
        # 1 diverse group
        self.begin("diverse")
        self.play(Write(f), run_time=1.4)
        B = group_bars(ax, div)
        self.play(Create(zero), FadeIn(B), FadeIn(names), run_time=1.4)
        st = MathTex(rf"\mu={m:.2f}\quad \sigma={s:.2f}", font_size=40, color=C_ST).to_corner(UR, buff=0.4).shift(DOWN * 1.2)
        self.play(FadeIn(st), run_time=0.8)
        self.say("a diverse group: some answers good, some bad")
        self.end()
        # 2 advantages
        self.begin("adv")
        A = [(v - m) / s for v in div]
        a1, p1 = morph(B, group_bars(ax, A, True)); self.play(*a1, run_time=2.2); finish(p1)
        self.say("clear winners (+) and losers (−) → something to push")
        self.end(2.2)
        # 3 identical rewards
        self.begin("identical")
        same = [1.0] * 8
        a1, p1 = morph(B, group_bars(ax, same)); self.play(*a1, run_time=2); finish(p1)
        st2 = MathTex(r"\mu=1.00\quad \sigma=0", font_size=40, color=C_ST).move_to(st)
        self.play(FadeTransform(st, st2), run_time=0.8); st = st2
        self.wait(0.6)
        a1, p1 = morph(B, group_bars(ax, [0.0] * 8, True)); self.play(*a1, run_time=2); finish(p1)
        self.say("all 8 correct: mean reward is perfect, every A = 0 → no gradient", C_BAD, 30)
        self.end(2.2)
        # 4 near-identical: group scaling amplifies noise
        self.begin("near")
        col = [0.86, 0.85, 0.85, 0.84, 0.85, 0.86, 0.85, 0.84]
        m2, s2 = stats(col)
        a1, p1 = morph(B, group_bars(ax, col)); self.play(*a1, run_time=1.8); finish(p1)
        st3 = MathTex(rf"\mu={m2:.2f}\quad \sigma={s2:.4f}", font_size=40, color=C_ST).move_to(st)
        self.play(FadeTransform(st, st3), run_time=0.8); st = st3
        self.say("almost identical rewards (0.84–0.86) …", C_AL)
        self.wait(0.8)
        sub = [v - m2 for v in col]
        a1, p1 = morph(B, group_bars(ax, sub, True)); self.play(*a1, run_time=1.6); finish(p1)
        self.say("subtract μ: differences of ±0.01", C_AL)
        self.wait(0.8)
        A2 = [(v - m2) / s2 for v in col]
        a1, p1 = morph(B, group_bars(ax, A2, True)); self.play(*a1, run_time=2); finish(p1)
        self.say("÷ tiny σ blows them up to ±1.3: learning from noise", C_BAD, 30)
        self.end(2.4)

# ---------------------------------------------------------------- L2 Chapter 2
BASE = np.array([2.2, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])
TOK = ["Paris", "The", "It", "France", "Lyon", "I", "Well", "As"]
def dist_T(T): return softmax(BASE / T)
def H(p): return float(-(p * np.log(p)).sum())
def T_for(h):
    lo, hi = 0.05, 1000.0
    for _ in range(100):
        mid = (lo * hi) ** 0.5
        if H(dist_T(mid)) > h: hi = mid
        else: lo = mid
    return mid

class Entropy(StepScene):
    def construct(self):
        f = MathTex(r"H", r"=", r"-\sum_{v}", r"\pi_\theta(v\mid\text{context})", r"\log", r"\pi_\theta(v\mid\text{context})", font_size=50)
        f[0].set_color(C_ENT); f[3].set_color(C_PI); f[5].set_color(C_PI)
        f.to_edge(UP, buff=0.4)
        ax = Axes(x_range=[0, 8, 1], y_range=[0, 1, 0.25], x_length=11, y_length=3.6).shift(DOWN * 0.7)
        base = Line(ax.c2p(0, 0), ax.c2p(8, 0), color=MUTED)
        names = VGroup(*[Text(t, font_size=26, color=C_Y).move_to(ax.c2p(i + 0.5, -0.12)) for i, t in enumerate(TOK)])
        def bars(p):
            g = VGroup()
            for i, v in enumerate(p):
                h = max(0.03, ax.c2p(0, v)[1] - ax.c2p(0, 0)[1])
                r = Rectangle(width=0.85, height=h, stroke_width=0, fill_color=C_PI if i == 0 else C_ENT, fill_opacity=0.9).move_to(ax.c2p(i + 0.5, 0), aligned_edge=DOWN)
                lab = Text(f"{v*100:.0f}%", font_size=24, color=FG).next_to(r, UP, buff=0.06)
                g.add(VGroup(r, lab))
            return g
        def htxt(p): return MathTex(rf"H={H(p):.2f}", font_size=48, color=C_ENT).to_corner(UR, buff=0.4).shift(DOWN * 1.3)
        # 1 formula
        self.begin("formula")
        self.play(Write(f[0:2]), run_time=1)
        self.play(Write(f[2:]), run_time=2)
        sub = Text("average uncertainty of the next token, in nats", font_size=28, color=MUTED).next_to(f, DOWN, buff=0.35)
        self.play(FadeIn(sub), run_time=0.8)
        self.end()
        # 2 flat
        self.begin("flat")
        p = np.ones(8) / 8
        B = bars(p); ht = htxt(p)
        self.play(FadeOut(sub), Create(base), FadeIn(names), FadeIn(B), FadeIn(ht), run_time=1.5)
        self.say("toy vocab of 8 tokens, flat: H = ln 8 = 2.08 (maximum)")
        self.end()
        # 3 sharpen to 1.08 then 0.88
        self.begin("sharpen")
        for h, msg in [(1.08, "early in your run: H ≈ 1.08"), (0.88, "later: H ≈ 0.88 → more peaked, more exploitative")]:
            p = dist_T(T_for(h))
            a1, p1 = morph(B, bars(p)); ht2 = htxt(p)
            self.play(*a1, FadeTransform(ht, ht2), run_time=2); finish(p1); ht = ht2
            self.say(msg, C_ENT if h > 1 else C_AL)
            self.wait(1.0)
        self.end()
        # 4 consequence for the group
        self.begin("group")
        self.play(FadeOut(VGroup(B, names, base, ht)), run_time=0.8)
        hdr = MathTex(r"P(\text{two samples pick the same token})=\sum_v \pi_\theta(v)^2", font_size=44).shift(UP * 1.2)
        self.play(Write(hdr), run_time=1.5)
        rows = VGroup()
        for h in (2.08, 1.08, 0.88):
            p = np.ones(8) / 8 if h > 2 else dist_T(T_for(h))
            rows.add(Text(f"H = {h:.2f}   →   {float((p**2).sum())*100:.0f}%", font_size=34, color=C_ENT if h > 1 else C_AL))
        rows.arrange(DOWN, buff=0.35).next_to(hdr, DOWN, buff=0.6)
        for r in rows:
            self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.8); self.wait(0.4)
        self.say("each token more predictable → group members more alike", FG, 30)
        self.end(2.2)

# ---------------------------------------------------------------- L2 Chapter 3
class Diagnose(StepScene):
    def construct(self):
        title = caption("Dashboard (illustrative curves)", 34).to_edge(UP, buff=0.3)
        x = np.linspace(0, 1, 200)
        curves = [
            ("mean reward", C_R, [0, 1], lambda t: 0.3 + 0.55 * (1 - np.exp(-6 * t)), "{:.2f}"),
            ("entropy", C_ENT, [0.8, 1.15], lambda t: 0.88 + 0.20 * np.exp(-4 * t), "{:.2f}"),
            ("reward_std (within group)", C_ST, [0, 0.4], lambda t: 0.04 + 0.31 * np.exp(-5 * t), "{:.2f}"),
            ("frac_reward_zero_std", C_BAD, [0, 1], lambda t: 0.1 + 0.75 * (1 - np.exp(-5 * t)), "{:.0%}"),
        ]
        panels = []
        for i, (name, col, yr, fn, fmt) in enumerate(curves):
            ax = Axes(x_range=[0, 1, 0.25], y_range=[yr[0], yr[1], (yr[1] - yr[0]) / 2], x_length=4.9, y_length=2.0,
                      axis_config={"color": MUTED, "include_tip": False, "include_ticks": False})
            lab = Text(name, font_size=26, color=col).next_to(ax, UP, buff=0.12)
            g = VGroup(ax, lab); panels.append((g, ax, col, fn, fmt))
        grid = VGroup(*[p[0] for p in panels]).arrange_in_grid(2, 2, buff=(1.9, 0.7)).next_to(title, DOWN, buff=0.35)
        def show(i, rt=2.0):
            g, ax, col, fn, fmt = panels[i]
            c = ax.plot(fn, x_range=[0, 1], color=col, stroke_width=5)
            end = Text(fmt.format(fn(1.0)), font_size=26, color=col).next_to(ax.c2p(1, fn(1.0)), RIGHT, buff=0.12)
            start = Text(fmt.format(fn(0.0)), font_size=24, color=MUTED).next_to(ax.c2p(0, fn(0.0)), LEFT, buff=0.12)
            self.play(FadeIn(g), run_time=0.6)
            self.play(Create(c), FadeIn(start), run_time=rt)
            self.play(FadeIn(end), run_time=0.5)
            return VGroup(c, end, start)
        # 1 mean reward
        self.begin("reward")
        self.play(FadeIn(title), run_time=0.6)
        show(0)
        self.say("mean reward climbs, then sits at ~0.85: looks fine…")
        self.end()
        # 2 entropy
        self.begin("entropy")
        show(1)
        self.say("entropy 1.08 → 0.88: more peaked (expected, up to a point)", C_ENT, 30)
        self.end()
        # 3 the real story
        self.begin("std")
        show(2, 1.6); show(3, 1.6)
        self.say("std → 0: most groups are all-equal → no gradient", C_BAD, 30)
        self.end(2.2)
        # 4 beta = 0
        self.begin("beta0")
        self.play(FadeOut(grid), FadeOut(title), *[FadeOut(m) for m in self.mobjects if m is not self.cap], run_time=0.8)
        self.unsay()
        rows = VGroup(
            Text("With β = 0, what still guards you?", font_size=36, color=FG),
            Text("✓  clipping: limits each step (does not add diversity)", font_size=30, color=C_GOOD),
            Text("✗  KL to a reference: off (β = 0), so no anchor", font_size=30, color=C_BAD),
            Text("→  watch entropy, reward_std, frac_reward_zero_std", font_size=30, color=C_AL),
            Text("     (not only mean reward)", font_size=30, color=C_AL),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to(ORIGIN)
        for r in rows:
            self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.8); self.wait(0.4)
        self.end(2.2)
