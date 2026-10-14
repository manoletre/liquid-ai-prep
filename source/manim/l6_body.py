# Lesson 6 · Surprise → cross-entropy → entropy → KL divergence (slow part 1 of the Lesson 4 split)
from fractions import Fraction as Fr
C_SIG = "#7aa2f7"; C_P = "#7aa2f7"; C_Q = "#ff9e64"; C_SUR = "#e0af68"; C_CE = "#e08fd0"; C_H = "#9ece6a"; C_KL = "#f7768e"; C_REF = "#7dcfff"

def fr(s): return float(Fr(s))

def dist_bars(ps, labels, color, w=0.8, gap=0.35, hmax=2.2, fs=22, hl=None, title=None):
    """Vertical probability bars; ps are fraction strings like '1/2'."""
    g = VGroup(); bars = VGroup()
    for i, (p, l) in enumerate(zip(ps, labels)):
        x = i * (w + gap); h = max(0.03, fr(p) * hmax)
        r = Rectangle(width=w, height=h, fill_color=color, fill_opacity=0.85, stroke_width=0).move_to([x, h / 2, 0])
        if hl == i: r.set_stroke(C_SUR, 4)
        val = Text(p, font_size=fs, color=FG).next_to(r, UP, buff=0.08)
        lab = Text(l, font_size=fs, color=C_SUR if hl == i else MUTED).move_to([x, -0.28, 0])
        base = Line([x - w / 2 - 0.08, 0, 0], [x + w / 2 + 0.08, 0, 0], color=GRID, stroke_width=2)
        bars.add(VGroup(base, r, val, lab))
    g.add(bars)
    if title:
        g.add(Text(title, font_size=fs + 2, color=color).next_to(bars, UP, buff=0.35))
    return g

def rows(*items, buff=0.3, size=28):
    out = VGroup()
    for it in items:
        if isinstance(it, tuple): out.add(Text(it[0], font_size=it[2] if len(it) > 2 else size, color=it[1]))
        else: out.add(Text(it, font_size=size))
    return out.arrange(DOWN, aligned_edge=LEFT, buff=buff)

# ---------------------------------------------------------------- Ch1: Recall
class Recall6(StepScene):
    def construct(self):
        title = caption("Recall first · three ideas from earlier lessons", 32).to_edge(UP, buff=0.35)
        # 1 PPO clip
        self.begin("clip")
        self.play(FadeIn(title), run_time=0.7)
        h = Text("PPO · good response (A > 0) · ρ = 1.3 · ε = 0.2", font_size=28, color=C_RHO).next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(h), run_time=0.7)
        ax = Axes(x_range=[0.5, 1.6, 0.1], y_range=[0, 1.5, 0.5], x_length=7.5, y_length=3.0,
                  axis_config={"color": MUTED, "stroke_width": 2}, tips=False).shift(DOWN * 0.55)
        xl = Text("ρ = π_θ / π_old", font_size=22, color=MUTED).next_to(ax, DOWN, buff=0.12)
        yl = Text("objective", font_size=22, color=MUTED).next_to(ax, LEFT, buff=0.12)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=1.0)
        f = ax.plot(lambda r: min(r, 1.2), x_range=[0.5, 1.6], color=C_PI, stroke_width=5)
        self.play(Create(f), run_time=1.4)
        e = DashedLine(ax.c2p(1.2, 0), ax.c2p(1.2, 1.45), color=C_EPS, stroke_width=3)
        el = Text("1 + ε = 1.2", font_size=22, color=C_EPS).next_to(e, UP, buff=0.08)
        dot = Dot(ax.c2p(1.3, 1.2), color=C_SUR, radius=0.1)
        dl = Text("ρ = 1.3: flat → gradient 0", font_size=24, color=C_SUR).next_to(dot, DR, buff=0.1)
        self.play(Create(e), FadeIn(el), run_time=0.8)
        self.play(FadeIn(dot), FadeIn(dl), run_time=0.8)
        self.say("past 1 + ε the clipped objective is flat: no extra push this batch", C_SUR, 26)
        self.end()
        # 2 GRPO equal rewards
        self.begin("grpo")
        self.play(FadeOut(VGroup(h, ax, xl, yl, f, e, el, dot, dl)), run_time=0.5)
        h2 = Text("GRPO · 8 answers to one prompt, all get reward 1", font_size=28, color=C_R).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(h2), run_time=0.7)
        boxes = VGroup(*[VGroup(Square(0.8, color=C_R, fill_opacity=0.15), Text("1", font_size=28, color=C_R)) for _ in range(8)])
        for b in boxes: b[1].move_to(b[0])
        boxes.arrange(RIGHT, buff=0.22).shift(UP * 0.6)
        self.play(FadeIn(boxes), run_time=1.0)
        m = Text("group mean μ = 1   →   A = R − μ = 0 for every answer", font_size=28, color=C_AL).shift(DOWN * 0.6)
        self.play(FadeIn(m), run_time=0.9)
        advs = VGroup(*[Text("A=0", font_size=20, color=MUTED).next_to(b, DOWN, buff=0.12) for b in boxes])
        self.play(FadeIn(advs), run_time=0.7)
        self.say("no difference inside the group → this prompt teaches nothing", C_AL, 28)
        self.end()
        # 3 logistic gradient
        self.begin("logreg")
        self.play(FadeOut(VGroup(h2, boxes, m, advs)), run_time=0.5)
        h3 = Text("Logistic regression · p = 0.8, true label y = 0, x = 2", font_size=28, color=C_SIG).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(h3), run_time=0.7)
        g = MathTex(r"\frac{\partial L}{\partial w}=(p-y)\,x=(0.8-0)\cdot 2=1.6", font_size=46).shift(UP * 0.5)
        self.play(Write(g), run_time=1.4)
        u = Text("positive gradient → w ← w − η·1.6 goes down → p goes down", font_size=26, color=C_GOOD).shift(DOWN * 0.9)
        self.play(FadeIn(u), run_time=0.8)
        self.say("p − y: how wrong the probability is, times the input", FG, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch2: Surprise
class Surprise(StepScene):
    def construct(self):
        title = caption("Surprise: what should an unlikely truth cost?", 32).to_edge(UP, buff=0.35)
        # 1 what we want
        self.begin("want")
        self.play(FadeIn(title), run_time=0.7)
        r = rows(("Your wake-word model said P(“Hey Guava”) = p. It WAS said.", FG, 28),
                 ("Rule 1 · p = 1 (certain and right) → cost 0", C_GOOD, 28),
                 ("Rule 2 · smaller p → bigger cost", C_SUR, 28),
                 ("Rule 3 · two independent clips → costs add", C_P, 28), buff=0.42).shift(UP * 0.2)
        for it in r:
            self.play(FadeIn(it, shift=RIGHT * 0.2), run_time=0.8); self.wait(0.4)
        self.say("we want a “surprise” number that obeys all three rules", FG, 28)
        self.end()
        # 2 halvings
        self.begin("halvings")
        self.play(FadeOut(r), run_time=0.5)
        ps = ["1", "1/2", "1/4", "1/8"]; bits = ["0", "1", "2", "3"]
        cols = VGroup()
        for p, b in zip(ps, bits):
            cols.add(VGroup(Text("p = " + p, font_size=30, color=C_P), Text(b + " bit" + ("" if b == "1" else "s"), font_size=30, color=C_SUR)).arrange(DOWN, buff=0.35))
        cols.arrange(RIGHT, buff=0.9).shift(UP * 1.0)
        hdr = Text("count the halvings from 1 down to p", font_size=26, color=MUTED).next_to(cols, UP, buff=0.35)
        self.play(FadeIn(hdr), run_time=0.6)
        for c in cols:
            self.play(FadeIn(c[0]), run_time=0.5); self.play(FadeIn(c[1]), run_time=0.5)
        f = MathTex(r"\text{surprise}(p)=-\log_2 p", font_size=52, color=C_SUR).shift(DOWN * 0.8)
        self.play(Write(f), run_time=1.2)
        ex = Text("p = 1/2 → 1 bit      p = 1/8 → 3 bits (three times the cost)", font_size=26, color=FG).shift(DOWN * 2.0)
        self.play(FadeIn(ex), run_time=0.8)
        self.say("−log₂ p = how many halvings of probability you were off by", C_SUR, 26)
        self.end()
        # 3 add
        self.begin("add")
        self.play(FadeOut(VGroup(hdr, cols, f, ex)), run_time=0.5)
        a = MathTex(r"P(\text{clip A and clip B}) = \tfrac12\cdot\tfrac14=\tfrac18", font_size=44).shift(UP * 1.3)
        self.play(Write(a), run_time=1.3)
        b = MathTex(r"-\log_2\tfrac18 = 3 \;=\; \underbrace{1}_{-\log_2 \frac12}+\underbrace{2}_{-\log_2 \frac14}", font_size=44, color=C_SUR).shift(DOWN * 0.3)
        self.play(Write(b), run_time=1.6)
        c = Text("log turns × (independent probabilities) into + (costs) → Rule 3", font_size=26, color=C_P).shift(DOWN * 1.9)
        self.play(FadeIn(c), run_time=0.8)
        self.say("this is why every loss you know has a log in it", FG, 28)
        self.end()
        # 4 edge + units + faded
        self.begin("edge")
        self.play(FadeOut(VGroup(a, b, c)), run_time=0.5)
        ax = Axes(x_range=[0, 1, 0.25], y_range=[0, 6, 1], x_length=5.4, y_length=3.4,
                  axis_config={"color": MUTED, "stroke_width": 2}, tips=False).shift(LEFT * 3.0 + DOWN * 0.35)
        xl = Text("p", font_size=24, color=MUTED).next_to(ax, DOWN, buff=0.12)
        cur = ax.plot(lambda p: -np.log2(p), x_range=[0.016, 1], color=C_SUR, stroke_width=5)
        self.play(Create(ax), FadeIn(xl), run_time=0.8)
        self.play(Create(cur), run_time=1.4)
        side = rows(("p → 0 · surprise → ∞", C_BAD, 28),
                    ("confident and wrong = huge cost", MUTED, 24),
                    ("ML code uses ln (nats)", C_P, 26),
                    ("1 bit = ln 2 ≈ 0.693 nats", MUTED, 24),
                    ("your turn: p = 1/16 → ? bits", C_AL, 28), buff=0.3).shift(RIGHT * 3.0 + DOWN * 0.1)
        self.play(FadeIn(side[:2]), run_time=0.8); self.wait(0.4)
        self.play(FadeIn(side[2:4]), run_time=0.8); self.wait(0.4)
        q = SurroundingRectangle(side[4], color=C_AL, buff=0.12)
        self.play(FadeIn(side[4]), Create(q), run_time=0.8)
        self.say("log base only changes the units, never which model wins", FG, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch3: Cross-entropy
class CrossEnt(StepScene):
    def construct(self):
        title = caption("Cross-entropy loss = average surprise at the true labels", 32).to_edge(UP, buff=0.35)
        # 1 likelihood
        self.begin("likelihood")
        self.play(FadeIn(title), run_time=0.7)
        hdr = Text("3 clips · intent classifier {yes, no, repeat}", font_size=26, color=MUTED).next_to(title, DOWN, buff=0.35)
        tbl = VGroup(
            VGroup(Text("clip", font_size=26, color=MUTED), Text("truth", font_size=26, color=MUTED), Text("model's prob on the truth", font_size=24, color=MUTED)),
            VGroup(Text("1", font_size=28), Text("yes", font_size=28, color=C_P), Text("1/2", font_size=28, color=C_Q)),
            VGroup(Text("2", font_size=28), Text("no", font_size=28, color=C_P), Text("1/4", font_size=28, color=C_Q)),
            VGroup(Text("3", font_size=28), Text("yes", font_size=28, color=C_P), Text("1/2", font_size=28, color=C_Q)),
        )
        xs = [-4.2, -1.6, 2.0]
        for r_, row in enumerate(tbl):
            for j, cell in enumerate(row): cell.move_to([xs[j], 1.6 - r_ * 0.62, 0])
        self.play(FadeIn(hdr), FadeIn(tbl[0]), run_time=0.7)
        for row in tbl[1:]: self.play(FadeIn(row), run_time=0.5)
        lik = MathTex(r"\text{likelihood}=\tfrac12\cdot\tfrac14\cdot\tfrac12=\tfrac1{16}", font_size=44, color=C_Q).shift(DOWN * 1.6)
        self.play(Write(lik), run_time=1.2)
        self.say("maximum likelihood: pick the model that makes the real labels most probable", FG, 24)
        self.end()
        # 2 log -> average surprise
        self.begin("log")
        self.play(FadeOut(lik), run_time=0.4)
        s = VGroup(*[Text(t, font_size=26, color=C_SUR).next_to(tbl[i + 1], RIGHT, buff=0.4) for i, t in enumerate(["1 bit", "2 bits", "1 bit"])])
        self.play(FadeIn(s), run_time=0.8)
        f1 = MathTex(r"-\log_2\tfrac1{16}=4 \text{ bits}=1+2+1", font_size=40, color=C_SUR).shift(DOWN * 1.3)
        self.play(Write(f1), run_time=1.2)
        f2 = MathTex(r"L_{CE}=-\frac1N\sum_i \log q(y_i)=\tfrac43\approx1.33\text{ bits/clip}", font_size=40, color=C_CE).shift(DOWN * 2.3)
        self.play(Write(f2), run_time=1.4)
        self.say("max likelihood = min total surprise = min cross-entropy", C_CE, 26)
        self.end()
        # 3 one-hot: only the true class counts
        self.begin("onehot")
        self.play(FadeOut(VGroup(hdr, tbl, s, f1, f2)), run_time=0.5)
        L = ["yes", "no", "repeat"]
        A = dist_bars(["1/2", "1/4", "1/4"], L, C_Q, hl=0, title="model A").shift(LEFT * 4.6 + DOWN * 0.9)
        B = dist_bars(["1/8", "1/2", "3/8"], L, C_Q, hl=0, title="model B").shift(RIGHT * 0.9 + DOWN * 0.9)
        tr = Text("true label: yes", font_size=26, color=C_SUR).next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(tr), FadeIn(A), FadeIn(B), run_time=1.2)
        la = Text("−log₂(1/2) = 1 bit", font_size=26, color=C_CE).next_to(A, DOWN, buff=0.35)
        lb = Text("−log₂(1/8) = 3 bits", font_size=26, color=C_CE).next_to(B, DOWN, buff=0.35)
        self.play(FadeIn(la), run_time=0.7); self.play(FadeIn(lb), run_time=0.7)
        self.say("one-hot label: −Σ y_k log q_k = −log q(true class); q = 0 → ∞", FG, 26)
        self.end()
        # 4 soft target -> general formula + faded
        self.begin("soft")
        self.play(FadeOut(VGroup(tr, A, B, la, lb)), run_time=0.5)
        f = MathTex(r"H(P,Q)=-\sum_k P(k)\,\log_2 Q(k)", font_size=46, color=C_CE).next_to(title, DOWN, buff=0.35)
        self.play(Write(f), run_time=1.2)
        Pb = dist_bars(["1/2", "1/4", "1/4"], ["a", "b", "c"], C_P, title="truth P", hmax=1.8).shift(LEFT * 5.0 + DOWN * 1.2)
        Qb = dist_bars(["1/4", "1/2", "1/4"], ["a", "b", "c"], C_Q, title="model Q", hmax=1.8).shift(LEFT * 1.2 + DOWN * 1.2)
        self.play(FadeIn(Pb), FadeIn(Qb), run_time=1.0)
        calc = rows(("½·2 + ¼·1 + ¼·2", FG, 28), ("= 1.75 bits", C_CE, 30), ("your turn: Q = P", C_AL, 26), ("½·1 + ¼·2 + ¼·2 = ?", C_AL, 26), buff=0.28).shift(RIGHT * 4.0 + DOWN * 0.6)
        self.play(FadeIn(calc[:2]), run_time=1.0); self.wait(0.4)
        q = SurroundingRectangle(calc[2:], color=C_AL, buff=0.12)
        self.play(FadeIn(calc[2:]), Create(q), run_time=0.9)
        self.say("weight each class's surprise by how often it is really the truth", FG, 26)
        self.end(2.2)

# ---------------------------------------------------------------- Ch4: Entropy
class Entropy(StepScene):
    def construct(self):
        title = caption("Entropy: the surprise you can't avoid", 32).to_edge(UP, buff=0.35)
        # 1 definition + fair coin
        self.begin("avg")
        self.play(FadeIn(title), run_time=0.7)
        f = MathTex(r"H(P)=H(P,P)=-\sum_k P(k)\log_2 P(k)", font_size=46, color=C_H).next_to(title, DOWN, buff=0.35)
        self.play(Write(f), run_time=1.3)
        sub = Text("predict with the true distribution itself — the best possible model", font_size=24, color=MUTED).next_to(f, DOWN, buff=0.25)
        self.play(FadeIn(sub), run_time=0.7)
        coin = dist_bars(["1/2", "1/2"], ["heads", "tails"], C_P, title="fair coin", hmax=1.8, w=1.1).shift(LEFT * 3.6 + DOWN * 1.6)
        self.play(FadeIn(coin), run_time=0.8)
        c = rows(("each side: −log₂ ½ = 1 bit", FG, 28), ("H = ½·1 + ½·1 = 1 bit", C_H, 30), buff=0.3).shift(RIGHT * 2.3 + DOWN * 0.9)
        self.play(FadeIn(c[0]), run_time=0.7); self.play(FadeIn(c[1]), run_time=0.7)
        self.say("worked example 1: a fair coin is worth exactly 1 bit", C_H, 28)
        self.end()
        # 2 three outcomes + uniform + faded
        self.begin("three")
        self.play(FadeOut(VGroup(sub, coin, c)), run_time=0.5)
        P = dist_bars(["1/2", "1/4", "1/4"], ["a", "b", "c"], C_P, title="P (from Ch 3)", hmax=1.8).shift(LEFT * 5.7 + DOWN * 1.5)
        U = dist_bars(["1/4", "1/4", "1/4", "1/4"], ["a", "b", "c", "d"], C_P, title="uniform over 4", hmax=1.8, w=0.6, gap=0.25).shift(LEFT * 0.3 + DOWN * 1.5)
        self.play(FadeIn(P), run_time=0.7)
        pc = Text("½·1 + ¼·2 + ¼·2 = 1.5 bits", font_size=24, color=C_H).next_to(P, DOWN, buff=0.3)
        self.play(FadeIn(pc), run_time=0.7)
        self.play(FadeIn(U), run_time=0.7)
        uc = Text("4 · (¼·2) = 2 bits = log₂ 4", font_size=24, color=C_H).next_to(U, DOWN, buff=0.3)
        self.play(FadeIn(uc), run_time=0.7)
        fd = rows(("your turn:", C_AL, 26), ("P = [½, ⅛, ⅛, ¼]", C_AL, 26), ("H = ?", C_AL, 26), buff=0.25).shift(RIGHT * 4.6 + DOWN * 0.4)
        q = SurroundingRectangle(fd, color=C_AL, buff=0.15)
        self.play(FadeIn(fd), Create(q), run_time=0.9)
        self.say("worked example 2: the 1.5 you just computed in Ch 3 was H(P)", FG, 26)
        self.end()
        # 3 edge case: certain
        self.begin("edge")
        self.play(FadeOut(VGroup(P, U, pc, uc, fd, q)), run_time=0.5)
        D = dist_bars(["1", "0", "0"], ["a", "b", "c"], C_P, title="certain", hmax=1.8).shift(LEFT * 4.8 + DOWN * 1.4)
        self.play(FadeIn(D), run_time=0.7)
        e = rows(("H = 1·log₂1 + 0 + 0 = 0 bits", C_H, 28), ("0·log 0 counts as 0 (x·log x → 0)", MUTED, 24),
                 ("peaked → low H     spread out → high H", FG, 26), ("max for K outcomes = log₂ K (uniform)", C_P, 26), buff=0.32).shift(RIGHT * 1.6 + DOWN * 0.7)
        for it in e: self.play(FadeIn(it), run_time=0.6); self.wait(0.3)
        self.say("edge case: nothing to be surprised about → entropy 0", C_H, 28)
        self.end()
        # 4 LLM link
        self.begin("llm")
        self.play(FadeOut(VGroup(D, e, f)), run_time=0.5)
        l = rows(("An LLM's next token is a distribution over 65,536 tokens", FG, 28),
                 ("TRL's entropy metric = average next-token entropy (nats)", C_H, 26),
                 ("your GRPO run: 1.08 → 0.88 = the policy got more peaked", C_SUR, 26),
                 ("too low → 8 samples look alike → equal rewards → A = 0 (Ch 1!)", C_BAD, 26), buff=0.4).shift(UP * 0.2)
        for it in l: self.play(FadeIn(it, shift=RIGHT * 0.2), run_time=0.8); self.wait(0.4)
        self.say("entropy is the exploration gauge in RL post-training", C_H, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch5: KL divergence
class KLDiv(StepScene):
    def construct(self):
        title = caption("KL divergence: the extra surprise from using the wrong model", 32).to_edge(UP, buff=0.35)
        # 1 recap bridge
        self.begin("recap")
        self.play(FadeIn(title), run_time=0.7)
        cards = VGroup()
        for t, f, c in [("surprise", r"-\log q", C_SUR), ("cross-entropy", r"H(P,Q)=-\sum P\log Q", C_CE), ("entropy", r"H(P)=-\sum P\log P", C_H)]:
            box = RoundedRectangle(width=4.0, height=1.9, corner_radius=0.15, stroke_color=c, fill_color=c, fill_opacity=0.08)
            g = VGroup(Text(t, font_size=26, color=c), MathTex(f, font_size=32, color=FG)).arrange(DOWN, buff=0.3).move_to(box)
            cards.add(VGroup(box, g))
        cards.arrange(RIGHT, buff=0.3).shift(UP * 0.5)
        for cd in cards: self.play(FadeIn(cd), run_time=0.7); self.wait(0.3)
        n = Text("same truth P = [½, ¼, ¼]:  H(P,Q) = 1.75 (Ch 3),  H(P) = 1.5 (Ch 4)", font_size=24, color=MUTED).shift(DOWN * 1.4)
        self.play(FadeIn(n), run_time=0.8)
        self.say("recap: what costs did we already compute for the same truth?", FG, 28)
        self.end()
        # 2 the gap
        self.begin("gap")
        self.play(FadeOut(VGroup(cards, n)), run_time=0.5)
        f = MathTex(r"D_{KL}(P\|Q)=H(P,Q)-H(P)=\sum_k P(k)\log_2\frac{P(k)}{Q(k)}", font_size=42, color=C_KL).next_to(title, DOWN, buff=0.4)
        self.play(Write(f), run_time=1.5)
        w1 = Text("way 1 · 1.75 − 1.5 = 0.25 bits", font_size=28, color=C_KL).shift(UP * 0.4)
        self.play(FadeIn(w1), run_time=0.8)
        w2 = VGroup(Text("way 2 · term by term", font_size=26, color=MUTED),
                    MathTex(r"\tfrac12\log_2\tfrac{1/2}{1/4}+\tfrac14\log_2\tfrac{1/4}{1/2}+\tfrac14\log_2\tfrac{1/4}{1/4}=0.5-0.25+0=0.25", font_size=40)).arrange(DOWN, buff=0.2).shift(DOWN * 0.75)
        self.play(Write(w2), run_time=1.8)
        nn = Text("one term is negative; the total never is (KL ≥ 0, = 0 only if Q = P)", font_size=24, color=MUTED).shift(DOWN * 2.05)
        self.play(FadeIn(nn), run_time=0.7)
        self.say("worked example 1: two ways, same 0.25 bits of wasted surprise", C_KL, 26)
        self.end()
        # 3 asymmetry edge case
        self.begin("asym")
        self.play(FadeOut(VGroup(f, w1, w2, nn)), run_time=0.5)
        P = dist_bars(["1", "0"], ["a", "b"], C_P, title="P", hmax=1.6, w=0.9).shift(LEFT * 5.6 + DOWN * 0.3)
        Q = dist_bars(["1/2", "1/2"], ["a", "b"], C_Q, title="Q", hmax=1.6, w=0.9).shift(LEFT * 3.0 + DOWN * 0.3)
        self.play(FadeIn(P), FadeIn(Q), run_time=0.9)
        a1 = MathTex(r"D_{KL}(P\|Q)=1\cdot\log_2\tfrac{1}{1/2}+0=1\text{ bit}", font_size=36, color=C_KL).shift(RIGHT * 2.4 + UP * 1.0)
        a2 = MathTex(r"D_{KL}(Q\|P)=\tfrac12\log_2\tfrac{1/2}{1}+\tfrac12\log_2\tfrac{1/2}{0}=\infty", font_size=36, color=C_BAD).shift(RIGHT * 2.4 + DOWN * 0.3)
        self.play(Write(a1), run_time=1.3); self.wait(0.4)
        self.play(Write(a2), run_time=1.5)
        a3 = Text("order matters → a divergence, not a distance (symmetric option: Jensen–Shannon)", font_size=22, color=MUTED).shift(DOWN * 2.0)
        self.play(FadeIn(a3), run_time=0.7)
        self.say("worked example 2 (edge case): Q puts mass where P has none → ∞", C_BAD, 26)
        self.end()
        # 4 training = minimizing KL
        self.begin("train")
        self.play(FadeOut(VGroup(P, Q, a1, a2, a3)), run_time=0.5)
        e = MathTex(r"\underbrace{H(P,Q)}_{\text{your loss}}=\underbrace{H(P)}_{\text{fixed by the data}}+\underbrace{D_{KL}(P\|Q)}_{\text{what training shrinks}}", font_size=44).shift(UP * 0.5)
        e[0][0:6].set_color(C_CE)
        self.play(Write(e), run_time=1.8)
        t = Text("so min cross-entropy = min KL to the data = max likelihood", font_size=28, color=C_KL).shift(DOWN * 1.5)
        self.play(FadeIn(t), run_time=0.8)
        self.say("classification training pulls Q toward the data's P", FG, 28)
        self.end()
        # 5 RLHF KL penalty + faded
        self.begin("rlhf")
        self.play(FadeOut(VGroup(e, t)), run_time=0.5)
        o = MathTex(r"\max_\theta\ \mathbb{E}[R]-\beta\,D_{KL}(\pi_\theta\|\pi_{\mathrm{ref}})", font_size=48).next_to(title, DOWN, buff=0.5)
        self.play(Write(o), run_time=1.4)
        r = rows(("per sampled token, estimate it with  log π_θ(y) − log π_ref(y)", C_REF, 26),
                 ("β > 0 → leash to the reference model (stay fluent, don't hack the reward)", C_H, 24),
                 ("β = 0 (your GRPO run) → no reference model, no leash", C_SUR, 24),
                 ("your turn: π_θ(y) = 1/2, π_ref(y) = 1/8 → log₂ ratio = ?", C_AL, 26), buff=0.35).shift(DOWN * 0.7)
        for it in r[:3]: self.play(FadeIn(it), run_time=0.7); self.wait(0.3)
        q = SurroundingRectangle(r[3], color=C_AL, buff=0.12)
        self.play(FadeIn(r[3]), Create(q), run_time=0.8)
        self.say("the β in PPO/GRPO configs weights exactly this KL", C_KL, 28)
        self.end(2.2)
