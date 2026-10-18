# Lesson 10 · Train/val/test, k-fold cross-validation, data leakage (last slow re-teach of Lesson 04)
C_TR = "#7aa2f7"; C_VA = "#e0af68"; C_TE = "#e08fd0"; C_LK = "#ff9e64"; C_OK = "#9ece6a"; C_NO = "#f7768e"
C_BP = "#ff9e64"; C_W10 = "#b392f0"; C_CL = "#f7768e"; C_AU = "#5fc9b6"

def rows10(*items, buff=0.3, size=28):
    out = VGroup()
    for it in items:
        if isinstance(it, tuple): out.add(Text(it[0], font_size=it[2] if len(it) > 2 else size, color=it[1]))
        else: out.add(Text(it, font_size=size))
    return out.arrange(DOWN, aligned_edge=LEFT, buff=buff)

def fit10(g, maxw=13.2):
    if g.width > maxw: g.scale_to_fit_width(maxw)
    return g

def seg_bar(parts, width=11.0, h=0.62, y=0.0, fs=24):
    """parts = [(label, frac, color)] → a horizontal bar split into labelled blocks."""
    g = VGroup(); x = -width / 2
    for lab, fr, col in parts:
        w = width * fr
        r = Rectangle(width=w, height=h, stroke_color="#0e1117", stroke_width=3, fill_color=col, fill_opacity=0.85).move_to([x + w / 2, y, 0])
        t = Text(lab, font_size=fs, color="#0e1117").move_to(r)
        if t.width > w - 0.1: t.scale_to_fit_width(max(w - 0.15, 0.2))
        g.add(VGroup(r, t)); x += w
    return g

def fold_grid(k, hi=None, width=9.0, h=0.42, y0=1.4, gap=0.14, labels=True):
    rows = VGroup()
    for i in range(k):
        row = VGroup()
        for j in range(k):
            col = C_VA if j == i else C_TR
            w = width / k
            r = Rectangle(width=w, height=h, stroke_color="#0e1117", stroke_width=3, fill_color=col,
                          fill_opacity=0.9 if (hi is None or hi == i) else 0.35).move_to([-width / 2 + w * (j + 0.5), y0 - i * (h + gap), 0])
            row.add(r)
        if labels:
            row.add(Text(f"run {i+1}", font_size=22, color=MUTED).next_to(row[0], LEFT, buff=0.25))
        rows.add(row)
    return rows

# ---------------------------------------------------------------- Ch1: Recall
class Recall10(StepScene):
    def construct(self):
        title = caption("Recall first · four ideas due today", 32).to_edge(UP, buff=0.35)
        # 1 backprop (Lesson 9, +1 day)
        self.begin("bp")
        self.play(FadeIn(title), run_time=0.7)
        h = Text("Lesson 9 · z = w₁x, h = ReLU(z), ŷ = w₂h, L = ½(ŷ − y)²", font_size=28, color=C_BP).next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(h), run_time=0.7)
        ex = Text("x = 2, w₁ = 1, w₂ = 3, y = 4", font_size=28, color=MUTED).next_to(h, DOWN, buff=0.3)
        self.play(FadeIn(ex), run_time=0.6)
        st = rows10(("forward: z = 2, h = 2, ŷ = 6, L = ½(6 − 4)² = 2", "#9ece6a", 26),
                    ("δŷ = ŷ − y = 2        ∂L/∂w₂ = δŷ · h = 2 · 2 = 4", C_BP, 26),
                    ("δh = δŷ · w₂ = 6     δz = 6 · 1 = 6 (ReLU on)", C_BP, 26),
                    ("∂L/∂w₁ = δz · x = 6 · 2 = 12", C_W10, 28),
                    ("edge: w₁ = −1 → z = −2, h = 0 → both gradients 0 (dead ReLU)", C_NO, 24), buff=0.28).shift(DOWN * 0.95)
        fit10(st)
        for it in st: self.play(FadeIn(it), run_time=0.65); self.wait(0.3)
        self.say("δ_in = δ_out × local slope;  weight gradient = δ × input", C_BP, 26)
        self.end()
        # 2 PPO clip, A < 0
        self.begin("clip")
        self.play(FadeOut(VGroup(h, ex, st)), run_time=0.5)
        h2 = Text("Lesson 1 · PPO clip, bad response A = −1, ε = 0.2", font_size=28, color=C_CL).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(h2), run_time=0.7)
        f = MathTex(r"\min\big(\rho A,\ \mathrm{clip}(\rho,\,0.8,\,1.2)\,A\big)", font_size=42).next_to(h2, DOWN, buff=0.35)
        self.play(Write(f), run_time=1.3)
        cl = rows10(("ρ = 0.9:  ρA = −0.9,  clip·A = −0.9  → min −0.9", C_OK, 27),
                    ("inside the range: not clipped, still pushed down", C_OK, 25),
                    ("ρ = 0.7:  ρA = −0.7,  clip·A = 0.8·(−1) = −0.8  → min −0.8", C_NO, 27),
                    ("the clipped term wins and is flat: gradient 0", C_NO, 25), buff=0.3).shift(DOWN * 1.1)
        fit10(cl)
        for it in cl: self.play(FadeIn(it), run_time=0.65); self.wait(0.3)
        self.say("for A < 0, the clip stops at 1 − ε: already made rare enough", C_CL, 26)
        self.end()
        # 3 AUROC with a tie
        self.begin("auroc")
        self.play(FadeOut(VGroup(h2, f, cl)), run_time=0.5)
        h3 = Text("Lesson 3 · AUROC: positives {0.7, 0.4}, negatives {0.4, 0.3, 0.1}", font_size=27, color=C_AU).next_to(title, DOWN, buff=0.4)
        fit10(h3); self.play(FadeIn(h3), run_time=0.7)
        au = rows10(("2 × 3 = 6 pairs (one positive vs one negative)", MUTED, 25),
                    ("0.7 beats 0.4, 0.3, 0.1   → 1 + 1 + 1 = 3", C_OK, 27),
                    ("0.4 vs 0.4 tie → ½,  beats 0.3 and 0.1 → 1 + 1", C_VA, 27),
                    ("AUROC = (3 + 2.5) / 6 = 5.5 / 6 ≈ 0.92", C_AU, 30), buff=0.34).shift(DOWN * 0.6)
        for it in au: self.play(FadeIn(it), run_time=0.65); self.wait(0.3)
        self.say("chance a random positive outscores a random negative; ties count ½", C_AU, 26)
        self.end()
        # 4 bias–variance → bridge to today
        self.begin("bv")
        self.play(FadeOut(VGroup(h3, au)), run_time=0.5)
        h4 = Text("Lesson 8 · train loss 0.05, validation loss 0.40 (made-up numbers)", font_size=27, color=C_VA).next_to(title, DOWN, buff=0.4)
        fit10(h4); self.play(FadeIn(h4), run_time=0.7)
        bv = rows10(("train low, big gap to validation → overfit (high variance)", C_VA, 27),
                    ("fixes: more data, stronger L2 / dropout, early stopping", C_OK, 26),
                    ("but: you could only SEE the gap because validation", FG, 26),
                    ("was data the model never trained on.  That's today.", FG, 26), buff=0.34).shift(DOWN * 0.6)
        for it in bv: self.play(FadeIn(it), run_time=0.65); self.wait(0.3)
        self.say("a held-out set is the only honest mirror", C_VA, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch2: Three splits
class Splits10(StepScene):
    def construct(self):
        title = caption("Train · validation · test: three jobs, three sets", 32).to_edge(UP, buff=0.35)
        # 1 why training score lies
        self.begin("why")
        self.play(FadeIn(title), run_time=0.7)
        q = Text("a student who memorized last year's practice exam", font_size=28, color=MUTED).shift(UP * 1.9)
        self.play(FadeIn(q), run_time=0.7)
        sc = rows10(("practice exam (seen before): 100%", C_TR, 30),
                    ("new exam (never seen): 55%", C_TE, 30), buff=0.4).shift(UP * 0.4)
        for it in sc: self.play(FadeIn(it), run_time=0.7); self.wait(0.4)
        m = rows10(("a model that memorizes its training data has 0 training error", FG, 26),
                   ("so the training score can't tell memorizing from learning", C_NO, 26), buff=0.3).shift(DOWN * 1.4)
        for it in m: self.play(FadeIn(it), run_time=0.7); self.wait(0.3)
        self.say("only data the model never saw can measure generalization", FG, 28)
        self.end()
        # 2 three jobs
        self.begin("jobs")
        self.play(FadeOut(VGroup(q, sc, m)), run_time=0.5)
        bar = seg_bar([("train 80%", 0.8, C_TR), ("val 10%", 0.1, C_VA), ("test 10%", 0.1, C_TE)], y=1.7)
        self.play(FadeIn(bar), run_time=1.0)
        jb = rows10(("train → fit the weights (gradient descent sees only this)", C_TR, 27),
                    ("validation → make choices: λ, learning rate, epochs, which model", C_VA, 27),
                    ("test → measure ONCE at the very end, then report it", C_TE, 27), buff=0.42).shift(DOWN * 0.4)
        fit10(jb)
        for it in jb: self.play(FadeIn(it), run_time=0.8); self.wait(0.5)
        self.say("validation is for deciding; test is for reporting", FG, 28)
        self.end()
        # 3 worked example 1: pick λ on val, report test
        self.begin("ex1")
        self.play(FadeOut(VGroup(bar, jb)), run_time=0.5)
        hd = Text("worked example 1 · pick λ (made-up error rates)", font_size=27, color=MUTED).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(hd), run_time=0.6)
        tb = rows10(("λ = 0      train 2%    val 15%", FG, 30),
                    ("λ = 0.1    train 6%    val  9%", FG, 30),
                    ("λ = 1      train 14%   val 16%", FG, 30), buff=0.35).shift(UP * 0.4)
        for it in tb: self.play(FadeIn(it), run_time=0.6); self.wait(0.2)
        box = SurroundingRectangle(tb[1], color=C_VA, buff=0.12)
        self.play(Create(box), run_time=0.7)
        rs = rows10(("lowest validation error → choose λ = 0.1", C_VA, 27),
                    ("then run the test set once: 10% → report 10%, not 9%", C_TE, 27), buff=0.3).shift(DOWN * 1.6)
        for it in rs: self.play(FadeIn(it), run_time=0.7); self.wait(0.4)
        self.say("val 9% is a bit flattering: you picked the best of three on it", C_VA, 26)
        self.end()
        # 4 worked example 2 (edge): choosing on the test set
        self.begin("ex2")
        self.play(FadeOut(VGroup(hd, tb, box, rs)), run_time=0.5)
        hd2 = Text("worked example 2 (edge) · 20 models that just flip coins", font_size=27, color=C_NO).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(hd2), run_time=0.6)
        rng = np.random.default_rng(3)
        acc = np.clip(0.5 + 0.035 * rng.standard_normal(20), 0.41, 0.565); acc[13] = 0.59
        ax = Axes(x_range=[0, 21, 5], y_range=[0.3, 0.7, 0.1], x_length=8.0, y_length=3.2, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2}).shift(DOWN * 0.15 + LEFT * 2.2)
        yl = VGroup(Text("50%", font_size=20, color=MUTED).next_to(ax.c2p(0, 0.5), LEFT, buff=0.15), Text("60%", font_size=20, color=MUTED).next_to(ax.c2p(0, 0.6), LEFT, buff=0.15), Text("test accuracy of each model", font_size=20, color=MUTED).next_to(ax, DOWN, buff=0.15))
        half = DashedLine(ax.c2p(0, 0.5), ax.c2p(21, 0.5), color=MUTED, stroke_width=2)
        bars = VGroup(*[Line(ax.c2p(i + 1, 0.3), ax.c2p(i + 1, a), color=C_TE if i != 13 else C_NO, stroke_width=10) for i, a in enumerate(acc)])
        self.play(Create(ax), FadeIn(yl), Create(half), run_time=1.0)
        self.play(LaggedStart(*[Create(b) for b in bars], lag_ratio=0.08), run_time=2.0)
        side = rows10(("100 test examples:", MUTED, 24),
                      ("each ≈ 50% ± 5%", FG, 24),
                      ("best of 20 ≈ 59%", C_NO, 26),
                      ("fresh data: 50%", C_OK, 26), buff=0.3).move_to([4.6, -0.1, 0])
        for it in side: self.play(FadeIn(it), run_time=0.6); self.wait(0.3)
        self.say("pick the winner ON the test set and the test score lies: it's become val", C_NO, 24)
        self.end()
        # 5 faded
        self.begin("faded")
        self.play(FadeOut(VGroup(hd2, ax, yl, half, bars, side)), run_time=0.5)
        hd3 = Text("your turn · model A and model B (made-up accuracies)", font_size=27, color=MUTED).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(hd3), run_time=0.6)
        tb2 = rows10(("model A    val 88%    test 85%", FG, 30),
                     ("model B    val 90%    test 83%", FG, 30), buff=0.4).shift(UP * 0.6)
        for it in tb2: self.play(FadeIn(it), run_time=0.6)
        st = rows10(("step 1: choose by validation → model B", C_VA, 27),
                    ("step 2: which number do you report?   ?", C_AL, 27), buff=0.35).shift(DOWN * 1.2)
        self.play(FadeIn(st[0]), run_time=0.7); self.wait(0.3)
        self.play(FadeIn(st[1]), Create(SurroundingRectangle(st[1], color=C_AL, buff=0.15)), run_time=0.8)
        self.say("tempting to switch to A after seeing test… don't", C_AL, 26)
        self.end(2.2)

# ---------------------------------------------------------------- Ch3: k-fold cross-validation
class KFold10(StepScene):
    def construct(self):
        title = caption("k-fold cross-validation · everyone takes a turn as the exam", 32).to_edge(UP, buff=0.35)
        # 1 why: one small val split is noisy
        self.begin("why")
        self.play(FadeIn(title), run_time=0.7)
        q = rows10(("only 50 labelled clips → hold out 10 for validation", MUTED, 27),
                   ("8 of 10 right = 80%  … but one clip flips it to 70% or 90%", C_VA, 27),
                   ("and 40 clips for training wastes a fifth of the data", C_NO, 27), buff=0.4).shift(UP * 0.7)
        fit10(q)
        for it in q: self.play(FadeIn(it), run_time=0.8); self.wait(0.4)
        self.say("one small validation split is a noisy, wasteful measurement", FG, 28)
        self.end()
        # 2 how it works
        self.begin("how")
        self.play(FadeOut(q), run_time=0.5)
        g = fold_grid(5, y0=1.8)
        lg = VGroup(Text("■ train on these 4 folds", font_size=24, color=C_TR), Text("■ validate on this fold", font_size=24, color=C_VA)).arrange(RIGHT, buff=0.8).move_to([0, -1.3, 0])
        for i, row in enumerate(g):
            self.play(FadeIn(row), run_time=0.55); self.wait(0.15)
        self.play(FadeIn(lg), run_time=0.6)
        ex = Text("5 trainings · each example is validated exactly once · average the 5 scores", font_size=24, color=FG).shift(DOWN * 2.35)
        fit10(ex); self.play(FadeIn(ex), run_time=0.8)
        self.say("k folds → k runs, each holding out a different fold", C_VA, 28)
        self.end()
        # 3 worked example 1: average
        self.begin("ex1")
        self.play(FadeOut(VGroup(lg, ex)), run_time=0.5)
        scores = [80, 85, 75, 90, 80]
        labs = VGroup(*[Text(f"{s}%", font_size=26, color=C_VA).next_to(g[i][-2], RIGHT, buff=1.0) for i, s in enumerate(scores)])
        self.play(LaggedStart(*[FadeIn(l) for l in labs], lag_ratio=0.3), run_time=1.6)
        cal = rows10(("worked example 1 · 100 examples, 5 folds of 20", MUTED, 24),
                     ("mean = (80 + 85 + 75 + 90 + 80) / 5 = 410 / 5 = 82%", C_VA, 27),
                     ("spread 75–90%: that's how noisy ONE split would have been", FG, 25), buff=0.25).shift(DOWN * 2.15)
        fit10(cal)
        for it in cal: self.play(FadeIn(it), run_time=0.7); self.wait(0.3)
        self.say("report the mean, and look at the spread", C_VA, 28)
        self.end()
        # 4 worked example 2 (edge): rare positives → stratify
        self.begin("ex2")
        self.play(FadeOut(VGroup(g, labs, cal)), run_time=0.5)
        hd = Text("worked example 2 (edge) · 100 clips, only 10 positives, 5 folds", font_size=27, color=C_NO).next_to(title, DOWN, buff=0.4)
        fit10(hd); self.play(FadeIn(hd), run_time=0.6)
        rnd = rows10(("random folds can land like: 4, 3, 3, 0, 0 positives", C_NO, 27),
                     ("a fold with 0 positives: recall = 0 / 0, undefined", C_NO, 27),
                     ("stratified: 10 ÷ 5 = 2 positives + 18 negatives per fold", C_OK, 27),
                     ("every fold keeps the 10% positive rate", C_OK, 25), buff=0.36).shift(DOWN * 0.3)
        fit10(rnd)
        for it in rnd: self.play(FadeIn(it), run_time=0.7); self.wait(0.35)
        self.say("with rare classes, use stratified k-fold", C_OK, 28)
        self.end()
        # 5 cost + after CV
        self.begin("cost")
        self.play(FadeOut(VGroup(hd, rnd)), run_time=0.5)
        cs = rows10(("cost: k trainings per setting you try", FG, 27),
                    ("4 values of λ × 5 folds = 20 trainings", C_VA, 27),
                    ("+ 1 final retrain on ALL the training data = 21", C_TR, 27),
                    ("then the untouched test set, once", C_TE, 27),
                    ("LLM fine-tuning: too costly → one fixed dev set instead", MUTED, 25), buff=0.34).shift(DOWN * 0.2)
        fit10(cs)
        for it in cs: self.play(FadeIn(it), run_time=0.7); self.wait(0.3)
        self.say("CV replaces the validation split, never the test set", FG, 28)
        self.end()
        # 6 faded
        self.begin("faded")
        self.play(FadeOut(cs), run_time=0.5)
        g4 = fold_grid(4, y0=1.6)
        s4 = [70, 74, 78, 74]
        l4 = VGroup(*[Text(f"{s}%", font_size=26, color=C_VA).next_to(g4[i][-2], RIGHT, buff=1.0) for i, s in enumerate(s4)])
        self.play(FadeIn(g4), run_time=0.9); self.play(FadeIn(l4), run_time=0.8)
        fd = Text("your turn · 4-fold mean = (70 + 74 + 78 + 74) / 4 = ?", font_size=28, color=C_AL).shift(DOWN * 1.6)
        self.play(FadeIn(fd), Create(SurroundingRectangle(fd, color=C_AL, buff=0.18)), run_time=0.9)
        self.say("add, then divide by k", C_AL, 28)
        self.end(2.2)

# ---------------------------------------------------------------- Ch4: Data leakage
class Leak10(StepScene):
    def construct(self):
        title = caption("Data leakage · when test information sneaks into training", 32).to_edge(UP, buff=0.35)
        # 1 definition
        self.begin("def")
        self.play(FadeIn(title), run_time=0.7)
        bar = seg_bar([("train", 0.8, C_TR), ("val", 0.1, C_VA), ("test", 0.1, C_TE)], y=1.5)
        self.play(FadeIn(bar), run_time=0.9)
        arr = CurvedArrow(bar[2].get_top() + UP * 0.05, bar[0].get_top() + UP * 0.05 + RIGHT * 1.5, angle=PI / 3, color=C_LK, stroke_width=5)
        self.play(Create(arr), run_time=1.2)
        d = rows10(("leakage = information the model couldn't have at prediction time", C_LK, 26),
                   ("(from val/test, from the future, or from the label itself)", MUTED, 24),
                   ("reaches training → the score looks great, then collapses in production", C_NO, 26), buff=0.3).shift(DOWN * 1.2)
        fit10(d)
        for it in d: self.play(FadeIn(it), run_time=0.75); self.wait(0.35)
        self.say("a held-out set only works if it was really held out", C_LK, 28)
        self.end()
        # 2 worked example 1: scaler fit on everything
        self.begin("ex1")
        self.play(FadeOut(VGroup(bar, arr, d)), run_time=0.5)
        hd = Text("worked example 1 · standardize a feature: z = (x − μ) / σ", font_size=27, color=MUTED).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(hd), run_time=0.6)
        ex = rows10(("train x = 2, 4, 6        test x = 12", FG, 28),
                    ("right: μ = (2 + 4 + 6) / 3 = 4  from train only", C_OK, 27),
                    ("σ = √((4 + 0 + 4) / 3) = √(8/3) ≈ 1.63", C_OK, 27),
                    ("test z = (12 − 4) / 1.63 ≈ 4.9   (an unusual value, honestly)", C_OK, 25),
                    ("leaky: μ = (2 + 4 + 6 + 12) / 4 = 6  — the test point moved μ", C_NO, 26), buff=0.3).shift(DOWN * 0.7)
        fit10(ex)
        for it in ex: self.play(FadeIn(it), run_time=0.7); self.wait(0.3)
        self.say("fit every preprocessing step on train only, then apply it to val/test", C_OK, 25)
        self.end()
        # 3 worked example 2 (edge): group leakage
        self.begin("ex2")
        self.play(FadeOut(VGroup(hd, ex)), run_time=0.5)
        hd2 = Text("worked example 2 (edge) · 5,000 clips from 50 speakers (made-up WERs)", font_size=26, color=C_NO).next_to(title, DOWN, buff=0.4)
        fit10(hd2); self.play(FadeIn(hd2), run_time=0.6)
        rs = VGroup(*[Rectangle(width=0.5, height=0.5, stroke_width=0, fill_color=c, fill_opacity=0.9) for c in [C_TR, C_TE, C_TR, C_TR, C_TE, C_TR, C_TR, C_TR]]).arrange(RIGHT, buff=0.1)
        sp = VGroup(*[Text(s, font_size=20, color="#0e1117").move_to(r) for s, r in zip("AABBCCDD", rs)])
        rl = Text("random split: speaker A in train AND test", font_size=24, color=C_NO).next_to(rs, RIGHT, buff=0.4)
        top = VGroup(rs, sp, rl).move_to([0, 1.0, 0]); fit10(top)
        self.play(FadeIn(top), run_time=0.9)
        gs = VGroup(*[Rectangle(width=0.5, height=0.5, stroke_width=0, fill_color=c, fill_opacity=0.9) for c in [C_TR, C_TR, C_TR, C_TR, C_TR, C_TR, C_TE, C_TE]]).arrange(RIGHT, buff=0.1)
        gp = VGroup(*[Text(s, font_size=20, color="#0e1117").move_to(r) for s, r in zip("AABBCCDD", gs)])
        gl = Text("group split: speaker D only in test", font_size=24, color=C_OK).next_to(gs, RIGHT, buff=0.4)
        bot = VGroup(gs, gp, gl).move_to([0, -0.1, 0]); fit10(bot)
        self.play(FadeIn(bot), run_time=0.9)
        res = rows10(("random split WER 6%  → model partly memorized each voice", C_NO, 26),
                     ("split by speaker WER 15%  → what new callers will get", C_OK, 26),
                     ("same for patients, users, documents, near-duplicate prompts", MUTED, 24), buff=0.3).shift(DOWN * 1.75)
        fit10(res)
        for it in res: self.play(FadeIn(it), run_time=0.7); self.wait(0.3)
        self.say("if examples come in groups, split by group (GroupKFold)", C_OK, 26)
        self.end()
        # 4 time + target leakage + LLM contamination
        self.begin("kinds")
        self.play(FadeOut(VGroup(hd2, top, bot, res)), run_time=0.5)
        kd = rows10(("time: shuffling a time series lets the past peek at the future", C_LK, 26),
                    ("   → train on earlier dates, test on later ones", C_OK, 24),
                    ("target: a feature that exists because of the label", C_LK, 26),
                    ("   (\"refund_issued\" when predicting fraud) → drop it", C_OK, 24),
                    ("LLMs: benchmark questions inside the training data", C_LK, 26),
                    ("   → dedupe by n-gram / near-duplicate search before training", C_OK, 24), buff=0.26).shift(DOWN * 0.2)
        fit10(kd)
        for it in kd: self.play(FadeIn(it), run_time=0.7); self.wait(0.25)
        self.say("ask: could the model know this at the moment it predicts?", FG, 27)
        self.end()
        # 5 faded: spot the leak
        self.begin("faded")
        self.play(FadeOut(kd), run_time=0.5)
        fd = rows10(("your turn · 1,000 random-noise features, 100 examples", MUTED, 25),
                    ("step 1: keep the 20 features most correlated with the label (all 100)", FG, 26),
                    ("step 2: 5-fold CV → 85% accuracy on pure noise", C_NO, 26),
                    ("where did the validation folds leak?   ?", C_AL, 27), buff=0.36).shift(UP * 0.1)
        fit10(fd)
        for it in fd[:-1]: self.play(FadeIn(it), run_time=0.7); self.wait(0.3)
        self.play(FadeIn(fd[-1]), Create(SurroundingRectangle(fd[-1], color=C_AL, buff=0.15)), run_time=0.8)
        self.say("look at which labels step 1 got to see", C_AL, 28)
        self.end(2.2)
