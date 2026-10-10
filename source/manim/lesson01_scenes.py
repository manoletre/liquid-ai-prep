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

# ---------------------------------------------------------------- (a)
class Reinforce(Scene):
    def construct(self):
        title = caption("REINFORCE: push up whatever you sampled", 36).to_edge(UP, buff=0.4)
        self.play(FadeIn(title))
        f = MathTex(r"\theta", r"\leftarrow", r"\theta", r"+", r"\alpha", r"\cdot", r"R", r"\cdot",
                    r"\nabla_\theta \log", r"\pi_\theta", r"(", r"y", r"\mid", r"x", r")", font_size=56)
        cols = {0: C_PI, 2: C_PI, 4: C_AL, 6: C_R, 9: C_PI, 11: C_Y, 13: C_X}
        for k, c in cols.items(): f[k].set_color(c)
        f.next_to(title, DOWN, buff=0.45)
        labels = [(4, "step size", C_AL), (6, "reward", C_R), (slice(8, 15), "make this response more likely", C_PI)]
        self.play(Write(f[0:4]), run_time=1)
        for sl, txt, c in labels:
            part = f[sl] if isinstance(sl, slice) else f[sl]
            lab = Text(txt, font_size=26, color=c).next_to(part, DOWN, buff=0.15)
            self.play(Write(part), FadeIn(lab, shift=UP * 0.1), run_time=1.0)
            self.play(FadeOut(lab), run_time=0.3)
        self.play(Write(f[5]), Write(f[7]), run_time=0.4)

        th = np.array(INIT, float)
        bars = Bars(softmax(th)).scale(0.95).move_to(DOWN * 1.15 + RIGHT * 0.2)
        prompt = Text('Prompt x: "What is the capital of France?"', font_size=30, color=C_X).next_to(bars, UP, buff=0.3)
        self.play(FadeIn(prompt), FadeIn(bars))
        note = caption("", 26)

        def step(th, y, alpha, R, msg, b=None, color=FG):
            coef = alpha * (R - (b or 0))
            th2 = reinforce(th, y, coef)
            hl = Bars(softmax(th), hl=y).scale(0.95).move_to(bars)
            new = Bars(softmax(th2), hl=y).scale(0.95).move_to(bars)
            m = caption(msg, 30, color).to_edge(DOWN, buff=0.3)
            a1, p1 = morph(bars, hl); self.play(*a1, FadeIn(m), run_time=0.6); finish(p1)
            self.wait(0.6)
            play_morph(self, bars, new, run_time=1.2)
            self.wait(1.0)
            self.play(FadeOut(m), run_time=0.3)
            return th2

        th = step(th, 1, 1.0, 0.8, "α = 1: sample “Paris.”, R = 0.8 → small push up")
        th = step(th, 0, 1.0, 1.0, "sample “Paris is the capital.”, R = 1.0 → push up")
        # reset, big step + unlucky sample
        th = np.array(INIT, float)
        reset = Bars(softmax(th)).scale(0.95).move_to(bars)
        warn = caption("Now: big step α = 8, and an unlucky sample", 28, C_AL).to_edge(DOWN, buff=0.35)
        a1, p1 = morph(bars, reset); self.play(*a1, FadeIn(warn)); finish(p1); self.wait(0.8); self.play(FadeOut(warn))
        th = step(th, 3, 8.0, 0.2, "sample “As an AI…”, R = 0.2 > 0 → still pushed up, hard", color=C_BAD)
        th = step(th, 3, 8.0, 0.2, "it's now likely, gets sampled again → collapse", color=C_BAD)
        # baseline
        f2 = MathTex(r"\theta", r"\leftarrow", r"\theta", r"+", r"\alpha", r"\cdot", r"(R-b)", r"\cdot",
                     r"\nabla_\theta \log", r"\pi_\theta", r"(", r"y", r"\mid", r"x", r")", font_size=56).move_to(f)
        for k, c in cols.items(): f2[k].set_color(c)
        f2[6].set_color(C_R)
        fix = caption("Fix: subtract a baseline b (typical reward). A = R − b", 28, C_R).to_edge(DOWN, buff=0.35)
        th = np.array(INIT, float)
        a1, p1 = morph(bars, Bars(softmax(th)).scale(0.95).move_to(bars)); self.play(TransformMatchingTex(f, f2), *a1, FadeIn(fix)); finish(p1)
        self.wait(1.2); self.play(FadeOut(fix))
        b = float(softmax(th) @ np.array(REW))
        th = step(th, 3, 8.0, 0.2, f"same sample: A = 0.2 − {b:.2f} = {0.2-b:+.2f} → pushed DOWN", b=b, color=C_GOOD)
        end = caption("Better than usual ↑   worse than usual ↓", 30, C_R).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(end)); self.wait(2)

# ---------------------------------------------------------------- (b)
class Clip(Scene):
    def construct(self):
        r = MathTex(r"\rho", r"=", r"{\pi_\theta(y\mid x)", r"\over", r"\pi_{\text{old}}(y\mid x)}", font_size=60)
        r[0].set_color(C_RHO); r[2].set_color(C_PI); r[4].set_color(C_OLD)
        self.play(Write(r[0:2]))
        self.play(Write(r[2]), FadeIn(Text("new policy", font_size=22, color=C_PI).next_to(r, RIGHT, buff=0.5).shift(UP * 0.4)))
        self.play(Write(r[3:]), FadeIn(Text("policy that sampled it", font_size=22, color=C_OLD).next_to(r, RIGHT, buff=0.5).shift(DOWN * 0.4)))
        ex = Text("ρ = 1.5  →  this response is now 50% more likely than when sampled", font_size=26, color=C_RHO).next_to(r, DOWN, buff=0.5)
        self.play(FadeIn(ex)); self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        L = MathTex(r"L", r"=", r"\min\big(", r"\rho", r"A", r",\;", r"\mathrm{clip}(", r"\rho", r",\,1-", r"\epsilon", r",\,1+", r"\epsilon", r")", r"A", r"\big)", font_size=48)
        for k, c in {3: C_RHO, 4: C_R, 7: C_RHO, 9: C_EPS, 11: C_EPS, 13: C_R}.items(): L[k].set_color(c)
        L.to_edge(UP, buff=0.35)
        self.play(Write(L[0:5])); self.play(Write(L[5:]))

        ax = Axes(x_range=[0, 2, 0.5], y_range=[-1.6, 1.6, 0.5], x_length=9.5, y_length=4.4,
                  axis_config={"color": MUTED, "include_tip": False}, x_axis_config={"numbers_to_include": [0.5, 1, 1.5, 2], "font_size": 26}).shift(DOWN * 0.45 + LEFT * 0.6)
        xl = MathTex(r"\rho", color=C_RHO, font_size=40).next_to(ax.x_axis, RIGHT, buff=0.2)
        self.play(Create(ax), FadeIn(xl))
        eps = 0.2
        tagA = MathTex(r"A=+1", color=C_R, font_size=44).to_corner(UR, buff=0.5).shift(DOWN * 1.3)
        self.play(FadeIn(tagA))
        unc = DashedVMobject(ax.plot(lambda x: x, x_range=[0, 2], color=MUTED), num_dashes=40)
        ul = Text("ρ·A (no clip)", font_size=26, color=MUTED).next_to(ax.c2p(1.9, 1.6), LEFT, buff=0.1)
        self.play(Create(unc), FadeIn(ul))
        vlines = VGroup(*[DashedLine(ax.c2p(v, -1.6), ax.c2p(v, 1.6), color=C_EPS, stroke_width=2) for v in (1 - eps, 1 + eps)])
        vl = VGroup(MathTex(r"1-\epsilon", color=C_EPS, font_size=30).next_to(vlines[0], UP, buff=0.08),
                    MathTex(r"1+\epsilon", color=C_EPS, font_size=30).next_to(vlines[1], UP, buff=0.08))
        self.play(Create(vlines), FadeIn(vl))
        def Lc(x, A): return min(x * A, np.clip(x, 1 - eps, 1 + eps) * A)
        cur = ax.plot(lambda x: Lc(x, 1), x_range=[0, 2, 0.005], color=C_R, stroke_width=6)
        self.play(Create(cur), run_time=1.5)
        zone = Rectangle(width=ax.c2p(2, 0)[0] - ax.c2p(1 + eps, 0)[0], height=ax.y_length, stroke_width=0, fill_color=C_EPS, fill_opacity=0.15)
        zone.move_to(ax.c2p((2 + 1 + eps) / 2, 0))
        zt = Text("flat → gradient = 0", font_size=28, color=C_EPS).move_to(ax.c2p(1.6, -0.6))
        self.play(FadeIn(zone), FadeIn(zt))
        t = ValueTracker(1.0)
        sgn = ValueTracker(1)
        dot = always_redraw(lambda: Dot(ax.c2p(t.get_value(), Lc(t.get_value(), 1 if sgn.get_value() > 0 else -1)), color=C_AL, radius=0.11))
        self.add(dot)
        self.play(t.animate.set_value(1.5), run_time=1.6)
        w = MathTex(r"\rho=1.5,\ A=+1:\ \min(1.5,\ 1.2)=1.2", r"\Rightarrow\ \nabla=0", font_size=38).to_edge(DOWN, buff=0.22)
        w[1].set_color(C_EPS)
        self.play(Write(w)); self.wait(1.2)
        self.play(FadeOut(w))
        s = caption("already 20% more likely → stop pushing", 32, C_EPS).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(s)); self.wait(1.5); self.play(FadeOut(s))

        # A < 0
        tagB = MathTex(r"A=-1", color=C_BAD, font_size=40).move_to(tagA)
        unc2 = DashedVMobject(ax.plot(lambda x: -x, x_range=[0, 1.6], color=MUTED), num_dashes=36)
        cur2 = ax.plot(lambda x: Lc(x, -1), x_range=[0, 1.6, 0.005], color=C_BAD, stroke_width=6)
        zone2 = Rectangle(width=ax.c2p(1 - eps, 0)[0] - ax.c2p(0, 0)[0], height=ax.y_length, stroke_width=0, fill_color=C_EPS, fill_opacity=0.15).move_to(ax.c2p((1 - eps) / 2, 0))
        zt2 = Text("flat → gradient = 0", font_size=28, color=C_EPS).move_to(ax.c2p(0.4, 0.6))
        self.play(Transform(tagA, tagB), Transform(unc, unc2), Transform(cur, cur2), Transform(zone, zone2), Transform(zt, zt2), FadeOut(ul), sgn.animate.set_value(-1), run_time=1.6)
        self.wait(0.8)
        w2 = MathTex(r"\rho=1.5,\ A=-1:\ \min(-1.5,\ -1.2)=-1.5", r"\Rightarrow\ \text{still active}", font_size=38).to_edge(DOWN, buff=0.22)
        w2[1].set_color(C_GOOD)
        self.play(Write(w2)); self.wait(1.2); self.play(FadeOut(w2))
        s2 = caption("a bad response got MORE likely → PPO always lets you undo it", 30, C_GOOD).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(s2))
        self.play(t.animate.set_value(0.5), run_time=2)
        s3 = caption("but once it's 20% less likely → stop", 32, C_EPS).to_edge(DOWN, buff=0.3)
        self.play(FadeOut(s2), FadeIn(s3)); self.wait(2.2)

# ---------------------------------------------------------------- (c)
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

class NaiveVsPPO(Scene):
    def construct(self):
        title = caption("Same batch, K = 10 gradient steps", 36).to_edge(UP, buff=0.4)
        sub = caption("pink tick = π_old   ·   ρ = π_θ / π_old", 24, MUTED).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title), FadeIn(sub))
        pold, A, out = run_epochs()
        def panel(th, kind):
            p = softmax(th)
            return Bars(p, old=pold, ratios=p / pold, width=2.7, fs=28, labels=SHORT).scale(0.85)
        hN = caption("Naive: maximize ρ·A", 32, FG); hP = caption("PPO: min(ρA, clip(ρ)A)", 32, FG)
        N = panel(out["naive"][0], "n").move_to(LEFT * 3.5 + DOWN * 0.55)
        P = panel(out["ppo"][0], "p").move_to(RIGHT * 3.5 + DOWN * 0.55)
        hN.next_to(N, UP, buff=0.35); hP.next_to(P, UP, buff=0.35)
        ep = Text("epoch 0", font_size=30, color=C_AL).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(N), FadeIn(P), FadeIn(hN), FadeIn(hP))
        intro = caption("16 responses sampled ONCE from π_old · A = R − batch mean", 28, C_R).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(intro)); self.wait(2.5); self.play(FadeOut(intro)); self.play(FadeIn(ep))
        for k in range(1, 11):
            N2 = panel(out["naive"][k], "n").move_to(N); P2 = panel(out["ppo"][k], "p").move_to(P)
            a1, p1 = morph(N, N2); a2, p2 = morph(P, P2); e2 = Text(f"epoch {k}", font_size=30, color=C_AL).move_to(ep)
            self.play(*a1, *a2, FadeTransform(ep, e2), run_time=1.0); finish(p1); finish(p2); ep = e2
        self.wait(0.8)
        mn = (softmax(out["naive"][-1]) / pold).max(); mp = (softmax(out["ppo"][-1]) / pold).max()
        vN = caption(f"max ρ = {mn:.1f}×: runs away", 30, C_BAD).next_to(N, DOWN, buff=0.45)
        vP = caption(f"max ρ = {mp:.2f}×: bounded", 30, C_GOOD).next_to(P, DOWN, buff=0.45)
        self.play(FadeOut(ep), FadeIn(vN), FadeIn(vP)); self.wait(3)

# ---------------------------------------------------------------- (d)
class GRPO(Scene):
    def construct(self):
        f = MathTex(r"A_i", r"=", r"{r_i", r"-", r"\mu", r"\over", r"\sigma}", font_size=60)
        f[0].set_color(C_R); f[2].set_color(C_R); f[4].set_color(C_ST); f[6].set_color(C_ST)
        f.to_edge(UP, buff=0.5)
        self.play(Write(f[0:3]))
        rew = [1, 0, 0, 1, 1, 0, 0, 0]
        ax = Axes(x_range=[0, 8, 1], y_range=[-1.6, 1.6, 0.5], x_length=12, y_length=4.6, axis_config={"color": MUTED, "include_tip": False}).shift(DOWN * 0.55)
        ax.x_axis.set_opacity(0.0)
        zero = Line(ax.c2p(0, 0), ax.c2p(8, 0), color=MUTED)
        def bars(vals, color_by_sign=False, prefix=""):
            g = VGroup()
            for i, v in enumerate(vals):
                c = C_R if not color_by_sign else (C_GOOD if v > 1e-9 else C_BAD if v < -1e-9 else MUTED)
                h = max(0.03, abs(ax.c2p(0, v)[1] - ax.c2p(0, 0)[1]))
                rct = Rectangle(width=0.8, height=h, stroke_width=0, fill_color=c, fill_opacity=0.9)
                rct.move_to(ax.c2p(i + 0.5, 0), aligned_edge=DOWN if v >= 0 else UP)
                lab = Text(f"{prefix}{v:.2f}".replace("-", "−"), font_size=26, color=FG).next_to(rct, UP if v >= 0 else DOWN, buff=0.08)
                g.add(VGroup(rct, lab))
            return g
        names = VGroup(*[MathTex(f"y_{i+1}", font_size=34, color=C_Y).move_to(ax.c2p(i + 0.5, -1.55)) for i in range(8)])
        B = bars(rew)
        cap = caption("8 responses to one prompt, 0/1 correctness", 32).to_edge(DOWN, buff=0.3)
        self.play(Create(zero), FadeIn(B), FadeIn(names), FadeIn(cap))
        self.wait(1)
        m = float(np.mean(rew)); s = float(np.std(rew, ddof=1))
        ml = DashedLine(ax.c2p(0, m), ax.c2p(8, m), color=C_ST)
        mt = MathTex(rf"\mu={m:.3f}", color=C_ST, font_size=34).next_to(ml.get_left(), UP + RIGHT, buff=0.1)
        def cap2(txt, col=FG): return caption(txt, 32, col).move_to(cap)
        n = cap2("group mean μ is the baseline (no critic needed)")
        self.play(Write(f[3:5]), Create(ml), FadeIn(mt), FadeTransform(cap, n)); cap = n
        self.wait(1)
        c = [v - m for v in rew]
        a1, p1 = morph(B, bars(c, True, "")); n = cap2("subtract μ: above average → positive, below → negative")
        self.play(*a1, ml.animate.move_to(ax.c2p(4, 0)), FadeOut(mt), FadeTransform(cap, n), run_time=1.5); finish(p1); cap = n
        self.wait(1)
        a = [v / s for v in c]
        a1, p1 = morph(B, bars(a, True, "")); n = cap2(f"divide by σ = {s:.2f}: these are the advantages A_i")
        self.play(Write(f[5:]), *a1, FadeTransform(cap, n), run_time=1.5); finish(p1); cap = n
        self.wait(2.5)
        a1, p1 = morph(B, bars([1] * 8)); n = cap2("Now all 8 are correct…")
        self.play(FadeOut(ml), *a1, FadeTransform(cap, n)); finish(p1); cap = n
        self.wait(1)
        a1, p1 = morph(B, bars([0] * 8, True, "")); n = cap2("σ = 0 → every A = 0 → no learning signal from this prompt", C_BAD)
        self.play(*a1, FadeTransform(cap, n), run_time=1.5); finish(p1); cap = n
        self.wait(3)
