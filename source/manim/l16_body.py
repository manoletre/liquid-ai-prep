
import json, os
STEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps")
HOLD = 1.8

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
    def tear_down(self):
        os.makedirs(STEP_DIR, exist_ok=True)
        with open(os.path.join(STEP_DIR, type(self).__name__ + ".json"), "w") as f:
            json.dump({"duration": round(self.renderer.time, 3), "steps": self.steps}, f, indent=1)

# Lesson 16 · GRPO reward scaling: scale_rewards = "group" / "batch" / "none"
C_T = "#e0af68"; C_OK = "#9ece6a"; C_NO = "#f7768e"; C_K = "#7aa2f7"; C_MP = "#5fc9b6"
C_R = "#9ece6a"; C_MU = "#c3e88d"; C_A = "#e0af68"; C_SG = "#ff9e64"; C_SB = "#7aa2f7"; C_RHO = "#ff9e64"; C_EPS = "#f7768e"
C_P = "#b392f0"; C_Q = "#5fc9b6"   # prompt P / prompt Q

def rows(*items, buff=0.3, size=28):
    out = VGroup()
    for it in items:
        if isinstance(it, tuple):
            out.add(Text(it[0], font_size=it[2] if len(it) > 2 else size, color=it[1]))
        else:
            out.add(Text(it, font_size=size))
    return out.arrange(DOWN, aligned_edge=LEFT, buff=buff)

def fit(g, maxw=13.2):
    if g.width > maxw: g.scale_to_fit_width(maxw)
    return g

class Base16(StepScene):
    def say(self, txt, color=FG, size=26, rt=0.8):
        new = fit(caption(txt, size, color), 13.0).to_edge(DOWN, buff=0.3)
        anims = [FadeIn(new, shift=UP * 0.1)]
        if self.cap is not None: anims.append(FadeOut(self.cap))
        self.play(*anims, run_time=rt); self.cap = new
    def title(self, s):
        self.ttl = fit(caption(s, 32), 13.0).to_edge(UP, buff=0.35); self.play(FadeIn(self.ttl), run_time=0.7); self.cur = None
    def clear_cur(self):
        if self.cur is not None: self.play(FadeOut(self.cur), run_time=0.5); self.cur = None
    def block(self, key, head, hc, lines, say, sc, hold=HOLD, box_last=False, buff=0.36):
        self.begin(key)
        self.clear_cur()
        h = Text(head, font_size=28, color=hc).next_to(self.ttl, DOWN, buff=0.4); fit(h)
        self.play(FadeIn(h), run_time=0.7)
        r = rows(*lines, buff=buff).next_to(h, DOWN, buff=0.45); fit(r)
        r.set_x(0)
        for it in r:
            self.play(FadeIn(it), run_time=0.75); self.wait(0.45)
        g = VGroup(h, r)
        if box_last:
            b = SurroundingRectangle(r[-1], color=C_T, buff=0.15); self.play(Create(b), run_time=0.7); g.add(b)
        self.say(say, sc, 26)
        self.end(hold)
        self.cur = g

def adv_panel(name, color, vals, unit=1.6, labels=None):
    """4 vertical advantage bars around a zero line; returns VGroup(label, zero, bars, nums)."""
    zero = Line(LEFT * 2.0, RIGHT * 2.0, color=MUTED, stroke_width=2)
    bars = VGroup(); nums = VGroup()
    for i, v in enumerate(vals):
        x = -1.5 + i * 1.0
        h = max(0.03, abs(v) * unit)
        b = Rectangle(width=0.55, height=h, stroke_width=0, fill_color=C_OK if v > 0 else (C_NO if v < 0 else MUTED), fill_opacity=0.9)
        b.move_to([x, (h / 2) * (1 if v >= 0 else -1), 0])
        bars.add(b)
        n = Text(f"{v:+.2f}" if v else "0", font_size=20, color=FG).move_to([x, (h + 0.25) * (1 if v >= 0 else -1), 0])
        nums.add(n)
    lab = Text(name, font_size=24, color=color).next_to(VGroup(bars, nums), UP, buff=0.25)
    return VGroup(lab, zero, bars, nums)

def panel_target(p, vals, unit=1.6):
    q = adv_panel(p[0].text, p[0].color, vals, unit)
    q.move_to(p, coor_mask=[1, 0, 0]); q.shift(UP * (p[1].get_y() - q[1].get_y()))
    return q

# ---------------------------------------------------------------- Ch1: Recall
class Recall16(Base16):
    def construct(self):
        self.title("Recall first · two ideas you've met before, then a bridge")
        self.block("clip", "PPO clip (Lessons 01, 07, 13) · ε = 0.2, so the safe range for ρ is 0.8 … 1.2", C_RHO, [
            ("objective = min( ρ·A ,  clip(ρ, 0.8, 1.2)·A )", FG, 26),
            ("A = +0.8, ρ = 1.25:  already pushed up past 1.2  →  clipped, gradient 0", C_NO, 24),
            ("A = +0.8, ρ = 0.6:  moved the WRONG way  →  not clipped, still pushed up", C_OK, 24),
            ("rule: the clip only stops you going further in the direction A wants", C_T, 24),
            ("A > 0: stops above 1 + ε.    A < 0: stops below 1 − ε", C_T, 24)],
            "the sign of A decides which side of the range is the fence", C_RHO)
        self.block("backprop", "backprop (Lesson 09) · x = 2,  h = ReLU(w₁x),  ŷ = w₂h,  L = ½(ŷ − y)²", C_K, [
            ("w₁ = 2, w₂ = 0.5, y = 3:   h = ReLU(4) = 4,   ŷ = 2,   L = ½(−1)² = 0.5", FG, 24),
            ("δŷ = ŷ − y = −1", C_SG, 25),
            ("∂L/∂w₂ = δŷ × h = −1 × 4 = −4", C_OK, 25),
            ("δh = δŷ × w₂ = −0.5;   ReLU is on (4 > 0), so it passes −0.5", C_SG, 24),
            ("∂L/∂w₁ = δ × x = −0.5 × 2 = −1", C_OK, 25)],
            "δ flows backward: multiply by local slopes, then δ × input", C_K)
        self.block("bridge", "Lesson 01 · the bridge to today", C_A, [
            ("GRPO samples G answers per prompt and scores each with a reward r", FG, 25),
            ("advantage A = r − (mean reward of that prompt's group)", C_A, 26),
            ("group [1, 0, 0, 0]:  mean 0.25  →  A = +0.75, −0.25, −0.25, −0.25", FG, 24),
            ("TRL then DIVIDES A by a spread σ.  scale_rewards picks which σ:", C_SG, 25),
            ("\"group\" (default)  ·  \"batch\" (your run)  ·  \"none\"", C_T, 25)],
            "and A is what PPO's clip multiplies, so its size is the size of the push", C_A, hold=2.2)

# ---------------------------------------------------------------- Ch2: group scaling
class Group16(Base16):
    def construct(self):
        self.title("scale_rewards = \"group\": divide by each prompt's own spread")
        # why: units
        self.begin("why")
        r = rows(("the push on an answer is proportional to its advantage A", FG, 26),
                 ("but A comes in the reward's units: a 0–1 reward vs a 0–10 reward = 10× the push", C_NO, 24),
                 ("everyday version: 80 on an exam where everyone got 78–82 is unremarkable;", MUTED, 24),
                 ("80 where everyone got 40–100 means something else again", MUTED, 24),
                 ("fix: measure “better than average” in units of the spread, not in points", C_OK, 26),
                 ("that's a z-score:  how many spreads above the mean", C_OK, 26), buff=0.34).next_to(self.ttl, DOWN, buff=0.6)
        fit(r)
        for it in r:
            self.play(FadeIn(it), run_time=0.75); self.wait(0.5)
        self.say("dividing by the spread removes the reward's units", C_OK)
        self.end(); self.cur = r
        # formula
        self.begin("formula")
        self.clear_cur()
        f = Text("A_i = ( r_i − μ_group ) / ( σ_group + 0.0001 )", font_size=40, color=C_A).next_to(self.ttl, DOWN, buff=0.6)
        self.play(FadeIn(f), run_time=1.2)
        r = rows(("μ_group = average reward of this prompt's G answers (the baseline, Lesson 01)", C_MU, 24),
                 ("σ_group = √( average of (r − μ)² )  — the typical distance from the mean", C_SG, 24),
                 ("+ 0.0001 only stops a divide-by-zero when all answers score the same", MUTED, 23),
                 ("TRL's torch .std() divides by G − 1, not G: a fixed factor per group size;", MUTED, 22),
                 ("by hand we divide by G, the pattern is the same", MUTED, 22), buff=0.32).next_to(f, DOWN, buff=0.55)
        fit(r)
        for it in r:
            self.play(FadeIn(it), run_time=0.75); self.wait(0.45)
        self.say("subtract the group's mean, divide by the group's own spread", C_A)
        self.end(); self.cur = VGroup(f, r)
        self.block("ex1", "worked example 1 · prompt P, G = 4, rewards [1, 1, 0, 0]", C_P, [
            ("μ = (1 + 1 + 0 + 0) / 4 = 0.5", C_MU, 26),
            ("r − μ = +0.5, +0.5, −0.5, −0.5", FG, 26),
            ("squares 0.25 each  →  average 0.25  →  σ = √0.25 = 0.5", C_SG, 26),
            ("A = ±0.5 / 0.5 = +1, +1, −1, −1", C_OK, 28)],
            "a 50/50 group: right answers +1, wrong answers −1", C_P)
        # ex2 with bars
        self.begin("ex2")
        self.clear_cur()
        h = Text("worked example 2 (edge cases) · prompt Q, partial credit [0.6, 0.6, 0.4, 0.4]", font_size=26, color=C_Q).next_to(self.ttl, DOWN, buff=0.35); fit(h)
        self.play(FadeIn(h), run_time=0.7)
        pP = adv_panel("P [1, 1, 0, 0]:  r − μ", C_P, [0.5, 0.5, -0.5, -0.5], unit=1.4).shift(LEFT * 3.3 + DOWN * 0.2)
        pQ = adv_panel("Q [.6, .6, .4, .4]:  r − μ", C_Q, [0.1, 0.1, -0.1, -0.1], unit=1.4).shift(RIGHT * 3.3 + DOWN * 0.2)
        self.play(FadeIn(pP), FadeIn(pQ), run_time=1.2); self.wait(1.0)
        self.say("before scaling: Q's differences are 5× smaller than P's (0.1 vs 0.5)", FG); self.wait(1.2)
        nP = panel_target(pP, [1, 1, -1, -1], 1.4); nQ = panel_target(pQ, [1, 1, -1, -1], 1.4)
        nP[0].become(Text("P:  ÷ σ = 0.5", font_size=24, color=C_P).move_to(nP[0])); nQ[0].become(Text("Q:  ÷ σ = 0.1", font_size=24, color=C_Q).move_to(nQ[0]))
        self.play(Transform(pP, nP), Transform(pQ, nQ), run_time=1.8); self.wait(0.8)
        self.say("after ÷ σ_group: Q gets the SAME ±1 push as P", C_NO); self.wait(1.0)
        e = Text("edge: [1, 1, 1, 1]  →  μ = 1, σ = 0  →  A = 0 / 0.0001 = 0;  TRL logs it in frac_reward_zero_std", font_size=22, color=C_T).next_to(pP, DOWN, buff=0.15).set_x(0); fit(e)
        self.play(FadeIn(e), run_time=0.8)
        self.say("group scaling blows small differences up to full size; all-equal groups give 0", C_NO)
        self.end(); self.cur = VGroup(h, pP, pQ, e)
        self.block("faded", "your turn · rewards [1, 0, 0, 0]", C_T, [
            ("μ = 0.25;   r − μ = +0.75, −0.25, −0.25, −0.25", C_MU, 25),
            ("squares 0.5625 + 3 × 0.0625 = 0.75  →  ÷ 4 = 0.1875  →  σ ≈ 0.433", C_SG, 25),
            ("A for the one correct answer = 0.75 / 0.433 = ?", C_T, 28)],
            "finish it", C_T, box_last=True, hold=2.2)

# ---------------------------------------------------------------- Ch3: batch scaling
class Batch16(Base16):
    def construct(self):
        self.title("scale_rewards = \"batch\": each prompt's own mean, one shared ruler")
        self.begin("idea")
        f = Text("A_i = ( r_i − μ_group ) / ( σ_batch + 0.0001 )", font_size=40, color=C_A).next_to(self.ttl, DOWN, buff=0.6)
        self.play(FadeIn(f), run_time=1.2)
        r = rows(("the baseline stays per prompt: μ_group, exactly as before", C_MU, 25),
                 ("only the ruler changes: ONE σ for the whole batch", C_SB, 26),
                 ("σ_batch = spread of all G × prompts rewards around the batch mean", C_SB, 24),
                 ("(TRL: rewards.std() over the whole batch)", MUTED, 22),
                 ("still unit-free, but a prompt with small differences keeps small advantages", C_OK, 24),
                 ("this is the “Lite PPO” recipe: group-level mean, batch-level std", MUTED, 22), buff=0.32).next_to(f, DOWN, buff=0.55)
        fit(r)
        for it in r:
            self.play(FadeIn(it), run_time=0.75); self.wait(0.45)
        self.say("same baseline, shared ruler", C_SB)
        self.end(); self.cur = VGroup(f, r)
        # ex1 with bars
        self.begin("ex1")
        self.clear_cur()
        h = Text("worked example 1 · P [1, 1, 0, 0] and Q [0.6, 0.6, 0.4, 0.4] in one batch", font_size=26, color=C_SB).next_to(self.ttl, DOWN, buff=0.35); fit(h)
        self.play(FadeIn(h), run_time=0.7)
        r = rows(("batch mean = 4.0 / 8 = 0.5;   distances: ±0.5 (×4), ±0.1 (×4)", FG, 23),
                 ("squares 4 × 0.25 + 4 × 0.01 = 1.04  →  ÷ 8 = 0.13  →  σ_batch ≈ 0.361", C_SB, 23), buff=0.22).next_to(h, DOWN, buff=0.3)
        fit(r); r.set_x(0)
        for it in r:
            self.play(FadeIn(it), run_time=0.75); self.wait(0.6)
        pP = adv_panel("P:  ±0.5 / 0.361", C_P, [1.386, 1.386, -1.386, -1.386], unit=0.75).shift(LEFT * 3.3 + DOWN * 1.15)
        pQ = adv_panel("Q:  ±0.1 / 0.361", C_Q, [0.277, 0.277, -0.277, -0.277], unit=0.75).shift(RIGHT * 3.3 + DOWN * 1.15)
        self.play(FadeIn(pP), FadeIn(pQ), run_time=1.4); self.wait(1.0)
        self.say("P ±1.39, Q ±0.28: the 5 : 1 ratio of the raw rewards survives", C_OK)
        self.end(); self.cur = VGroup(h, r, pP, pQ)
        self.block("ex2", "worked example 2 (edge) · batch = [1, 1, 1, 1] + [1, 1, 0, 0]", C_NO, [
            ("batch mean = 6 / 8 = 0.75", FG, 25),
            ("squares 6 × 0.0625 + 2 × 0.5625 = 1.5  →  ÷ 8 = 0.1875  →  σ_batch ≈ 0.433", C_SB, 24),
            ("group 1: μ = 1, every r − μ = 0  →  A = 0 (still no learning signal)", C_MU, 24),
            ("group 2: μ = 0.5  →  A = ±0.5 / 0.433 = ±1.15", C_OK, 25),
            ("frac_reward_zero_std tests σ_batch = 0.433 ≠ 0  →  logs 0", C_NO, 25),
            ("…even though half the batch had zero spread", C_NO, 25)],
            "your run's 0.000 for 600 steps: in batch mode that metric can't see dead groups", C_NO)
        self.block("faded", "your turn · batch = [1, 0, 0, 0] + [0, 0, 0, 0]", C_T, [
            ("batch mean = 1 / 8 = 0.125", FG, 26),
            ("squares 0.875² + 7 × 0.125² = 0.875  →  ÷ 8 = 0.109  →  σ_batch ≈ 0.331", C_SB, 24),
            ("group 1: μ = 0.25.   A for the correct answer = 0.75 / 0.331 = ?", C_T, 27)],
            "finish it", C_T, box_last=True, hold=2.2)

# ---------------------------------------------------------------- Ch4: choosing
class Choose16(Base16):
    def construct(self):
        self.title("group, batch or none? Read the effect on easy and hard prompts")
        self.block("bridge", "recap: one baseline, three rulers", C_A, [
            ("always: A = r − μ_group   (the prompt's own average is the baseline)", C_MU, 25),
            ("\"group\":  ÷ σ of that prompt's answers   →  every prompt gets the same-size push", C_SG, 24),
            ("\"batch\":  ÷ one σ for the whole batch    →  sizes follow the raw reward gaps", C_SB, 24),
            ("\"none\":   no division                   →  A in reward units (Dr. GRPO)", C_T, 24)],
            "now: what does each do with 0/1 correctness rewards?", C_A)
        self.block("binary", "where σ comes from for 0/1 rewards (fraction correct = p)", C_SG, [
            ("mean = p   (a share p of answers score 1, the rest 0)", C_MU, 25),
            ("right answers sit (1 − p) above the mean, wrong ones p below it", FG, 25),
            ("variance = p·(1 − p)² + (1 − p)·p² = p(1 − p)·[(1 − p) + p] = p(1 − p)", C_SG, 24),
            ("σ = √( p(1 − p) ):  biggest at p = ½ (σ = 0.5), tiny near p = 0 or 1", C_SG, 25),
            ("so dividing by σ_group BOOSTS nearly-solved and nearly-impossible prompts", C_NO, 24)],
            "σ is small exactly when a prompt is too easy or too hard", C_SG)
        self.block("ex1", "worked example 1 · G = 8:  easy prompt E 7/8 correct,  medium M 4/8", C_K, [
            ("σ_E = √(0.875 × 0.125) ≈ 0.331      σ_M = √(0.5 × 0.5) = 0.5", C_SG, 25),
            ("a wrong answer, r − μ:   E  −0.875     M  −0.5", FG, 25),
            ("\"group\":   E  −0.875 / 0.331 = −2.65     M  −0.5 / 0.5 = −1.00", C_NO, 25),
            ("\"batch\":   σ of all 16 ≈ 0.464  →  E  −1.89     M  −1.08", C_SB, 25),
            ("\"none\":    E  −0.875     M  −0.5", C_T, 25)],
            "group: E's mistake gets 2.65× M's push; batch and none keep the raw 1.75×", C_K)
        self.block("ex2", "worked example 2 (edge) · a tiny format bonus under \"group\"", C_NO, [
            ("reward = correctness (0/1) + 0.01 if the format is perfect", FG, 25),
            ("an all-correct group: [1.01, 1.00, 1.01, 1.00]  →  μ = 1.005, σ = 0.005", C_SG, 24),
            ("A = ±0.005 / (0.005 + 0.0001) ≈ ±0.98", C_NO, 26),
            ("a 0.01 formatting bonus now pushes as hard as being right vs wrong elsewhere", C_NO, 24),
            ("\"batch\": σ_batch ≈ 0.4–0.5 if other prompts mix 0s and 1s  →  A ≈ ±0.01", C_OK, 24)],
            "group scaling amplifies whatever little differences a group has, even noise", C_NO)
        self.block("choose", "how to choose (and what to log)", C_MP, [
            ("\"group\" (default): equal push per prompt; amplifies near-solved, noisy groups", C_SG, 24),
            ("\"batch\" (Lite PPO, your run): one stable ruler, keeps real gap sizes", C_SB, 24),
            ("\"none\" (Dr. GRPO): no difficulty bias; lr must suit the reward scale", C_T, 24),
            ("all three: a group whose answers all score the same gets A = 0", C_MU, 24),
            ("batch mode: log your own share of groups with σ_group = 0", C_OK, 24)],
            "no universal winner: try it on your eval; know what each one amplifies", C_MP)
        self.block("faded", "your turn · \"group\" mode, G = 8, prompt with 6/8 correct", C_T, [
            ("p = 0.75,  σ = √(0.75 × 0.25) ≈ 0.433", C_SG, 27),
            ("a wrong answer: r − μ = 0 − 0.75 = −0.75", FG, 27),
            ("A = −0.75 / 0.433 = ?", C_T, 29)],
            "finish it", C_T, box_last=True, hold=2.2)
